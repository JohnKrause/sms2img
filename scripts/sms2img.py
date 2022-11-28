import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from twilio.rest import Client

import json

torch.set_grad_enabled(False)

import time
from multiprocessing import Process, Pipe
from flask import Flask, request, send_from_directory
import random

from contextlib import redirect_stderr


class request_content():
    def __init__(self):
        self.action=""
        self.id=0
        self.phone=""
        self.file_urls=[]
        self.verify_num=0
        self.q_pos=0
        self.sms_data=0
        self.prompt=""
        self.neg_prompt=""
        self.images=[]
    def new_id(self):
        self.id = random.randint(0,1000000000000000000)
    def new_verify_num(self):
        self.verify_num = random.randint(0,1000)
    def toJson(self):
        return {"action":self.action,"id":self.id,"phone":self.phone,"file_urls":self.file_urls,"verify_num":self.verify_num,"q_pos":self.q_pos,"sms_data":self.sms_data,"prompt":self.prompt}

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def handle_sms_request(request_content, client_db):
    return_value=False
    if(request_content.action=="exit"):
        return_value=True
    elif(request_content.action=="analyze_sms"):
        if(not client_db_known(request_content,client_db)):
            if(client_db_enroll(request_content,client_db)):
                client_db_new_client(request_content,client_db)
                request_content.action="new_user_start"
                return_value=True
            else:
                return_value=False
        else:
            if(client_db_flood(request_content,client_db)):
                return_value=False

            elif(client_db_verified(request_content,client_db)):
                if(client_db_allow_new_prompt(request_content,client_db)):
                    request_content.action="prompt_rcvd"
                else:
                    request_content.action="prompt_overflow"
                return_value=True

            elif(client_db_verify(request_content,client_db)):
                request_content.action="new_user_success"
                return_value=True

            else:
                return_value=False

            client_db_update_flood(request_content,client_db)
    return return_value

def handle_gpu_request(request_content, client_db):
    client_db[request_content.phone]["in_q"]-=1
    request_content.new_id()
    file_urls=[]
    for num,img in enumerate(request_content.images):
        filename=f"results/{request_content.id}_{num}.png"                     
        img.save("scripts/"+filename)
        file_urls.append("http://krauseml.crabdance.com:5000/"+filename)
    request_content.file_urls = file_urls
    request_content.action="prompt_finished"
    return True

def client_db_load():
    with open('client_db.json','r+') as jsonfile:
        client_db=json.load(jsonfile)
    return client_db

def client_db_save(client_db):
    with open('client_db.json', "r") as jsonfile_current:
        client_db_old=json.load(jsonfile_current)
    with open(f'db_backups/client_db_back_{int(time.time())}.json', "w") as jsonfile_backup:
        json.dump(client_db_old,jsonfile_backup)
    with open('client_db.json', "w") as jsonfile_current:
        json.dump(client_db, jsonfile_current)

def client_db_new_client(request_content, client_db):
    request_content.new_verify_num()
    client_db[request_content.phone]={"status":"unverified", "in_q":0, "num_requests":0, "verify_num":f'{request_content.verify_num}', "last_seen":int(time.time())}

def client_db_verify(request_content, client_db):
    if(request_content.prompt == client_db[request_content.phone]["verify_num"]):
        client_db[request_content.phone]["status"]="verified"
        return True
    else:
        client_db[request_content.phone]["status"]="unverified"
        return False

def client_db_verified(request_content, client_db):
    return (client_db[request_content.phone]["status"]=="verified")

def client_db_known(request_content, client_db):
    return (request_content.phone in client_db)

def client_db_allow_new_prompt(request_content, client_db):
    if(client_db[request_content.phone]["in_q"]<3):
        client_db[request_content.phone]["num_requests"]+=1
        client_db[request_content.phone]["in_q"]+=1
        return True
    else:
        return False

def client_db_flood(request_content, client_db):
    return (client_db[request_content.phone]["last_seen"]>time.time())

def client_db_update_flood(request_content, client_db):
    client_db[request_content.phone]["last_seen"]=(time.time()+1)

def client_db_enroll(request_content, client_db):
    return (request_content.prompt.lower().strip()=="enroll")

app = Flask(__name__)
@app.route('/inbound', methods=['POST'])
def inbound():
    global g_pipe_route
    content=request_content()
    if(len(request.values.get('Body', None))<500 and len(request.values.get('Body', None))>0):
        content=request_content()
        content.action="analyze_sms"
        content.sms_data=request.get_data()
        full_prompt = request.values.get('Body', None)
        if ("@" in full_prompt):
            content.prompt = full_prompt.split("@")[0]
            content.neg_prompt = full_prompt.split("@")[1]
        else:
            content.prompt = full_prompt
            content.neg_prompt = ""
        content.phone = request.values.get('From', None)
        g_pipe_route.send(content)
    return ""

@app.route('/results/<path:filename>')
def results(filename):
    return send_from_directory('results', filename, as_attachment=True)

def gpu_process(pipe):
    opt_ckpt="sd2/768-v-ema.ckpt"
    opt_seed=420
    opt_config="configs/stable-diffusion/v2-inference-v.yaml"
    opt_plms=False
    opt_dpm=False
    opt_n_samples=1
    opt_n_rows=0
    opt_C=4
    opt_H=768
    opt_f=8
    opt_W=768
    opt_steps=40
    opt_scale=9.0
    opt_ddim_eta=0.0
    opt_n_iter=2

    seed_everything(opt_seed)

    config = OmegaConf.load(f"{opt_config}")
    model = load_model_from_config(config, f"{opt_ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt_plms:
        sampler = PLMSSampler(model)
    elif opt_dpm:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

    batch_size = opt_n_samples
    n_rows = opt_n_rows if opt_n_rows > 0 else batch_size

    start_code=None
    precision_scope = autocast

    with torch.no_grad(), \
        precision_scope("cuda"), \
        model.ema_scope():
            run=True
            while(run):
                if(pipe.poll(1)):
                    request_content=pipe.recv()
                    try:
                        if(request_content.action=="exit"):
                            run=False
                            break
                        elif(request_content.action=="prompt_rcvd"):
                            print(f'\x1b[36m Running computation for {request_content.phone}')
                            data=[[request_content.prompt]]
                            all_imgs = list()
                            for n in trange(opt_n_iter, desc="Sampling", disable=True):
                                for prompts in tqdm(data, desc="data", disable=True):
                                    uc = None
                                    if opt_scale != 1.0:
                                        uc = model.get_learned_conditioning([request_content.neg_prompt])
                                    if isinstance(prompts, tuple):
                                        prompts = list(prompts)
                                    c = model.get_learned_conditioning(prompts)
                                    shape = [opt_C, opt_H // opt_f, opt_W // opt_f]
                                    samples, _ = sampler.sample(S=opt_steps,
                                                                     conditioning=c,
                                                                     batch_size=opt_n_samples,
                                                                     shape=shape,
                                                                     verbose=False,
                                                                     unconditional_guidance_scale=opt_scale,
                                                                     unconditional_conditioning=uc,
                                                                     eta=opt_ddim_eta,
                                                                     x_T=start_code)
                                    x_samples = model.decode_first_stage(samples)
                                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                                    for x_sample in x_samples:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                        img = Image.fromarray(x_sample.astype(np.uint8))
                                        all_imgs.append(img)
                        else:
                            continue
                        request_content.images=all_imgs
                        pipe.send(request_content)
                    except Exception as e:
                        print(e)

def flask_process():
    app.run(host='0.0.0.0')
    while(True):
        time.sleep(1)

def sms_send_process(pipe,sid,auth):
    run=True 
    client = Client(sid,auth)
    while(run):
        if(pipe.poll(1)):
            request_content = pipe.recv()
            try:
                if(request_content.action=="exit"):
                    run=False
                    break
                elif(request_content.action=="prompt_finished"):
                    message = client.messages.create(
                        to=request_content.phone,
                        from_="+15139958545",
                        media_url=request_content.file_urls,
                        body="")
                    print(f'\x1b[31m Sent finished prompt to: {request_content.phone}, {message.sid}')
                elif(request_content.action=="new_user_start"):
                    message = client.messages.create(
                        to=request_content.phone,
                        from_="+15139958545",
                        body=f"Welcome to SMS-2-Image! To begin please validate by replying with the following code:{request_content.verify_num}")
                    print(f'\x1b[32m Sent Welcome message to: {request_content.phone}, {message.sid}')
                elif(request_content.action=="new_user_success"):
                    message = client.messages.create(
                        to=request_content.phone,
                        from_="+15139958545",
                        body=f"Success! You may now send image prompts. For example:'Photograph of an astronaut riding a horse'. Results will be different everytime you try!")
                    print(f'\x1b[33m New user success: {request_content.phone}, {message.sid}')
                elif(request_content.action=="prompt_rcvd"):
                    message = client.messages.create(
                        to=request_content.phone,
                        from_="+15139958545",
                        body=f"Prompt Received, Position in queue:{request_content.q_pos}")
                    print(f'\x1b[34m Prompt Received from: {request_content.phone}, {message.sid}')
                elif(request_content.action=="prompt_overflow"):
                    message = client.messages.create(
                        to=request_content.phone,
                        from_="+15139958545",
                        body=f"Too many prompts in Queue, Please wait")
                    print(f'\x1b[35m Overflow message: {request_content.phone}, {message.sid}')
                else:
                    continue
            except Exception as e:
                print(e)
                continue

def main():

    print("starting the gpu process")
    pipe_gpu, pipe_gpu_local = Pipe()
    gpu_proc=Process(target=gpu_process,args=(pipe_gpu,))
    gpu_proc.start()

    print("starting webserver process")
    global g_pipe_route
    pipe_route, pipe_route_local = Pipe()
    g_pipe_route=pipe_route
    flask_proc=Process(target=flask_process)
    flask_proc.start()

    #time.sleep(10)

    print("starting sms sending process")
    account_sid=input("account sid? ")
    auth_token=input("auth token? ")
    pipe_sms_send, pipe_sms_send_local = Pipe()
    sms_send_proc=Process(target=sms_send_process,args=(pipe_sms_send,account_sid,auth_token))
    sms_send_proc.start()

    print("loading client db")
    client_db=client_db_load()
    db_save_delay=60*60*3
    next_db_save=time.time()+db_save_delay

    queue_size=0
    request_id=0
    run=True
    try:
        while(run):
            if(pipe_route_local.poll(0.1)):
                try:
                    request_content=pipe_route_local.recv()
                    if(handle_sms_request(request_content, client_db)):
                        if(request_content.action=="exit"):
                            run=False
                            break
                        elif(request_content.action=="new_user_start"):
                            pipe_sms_send_local.send(request_content)
                        elif(request_content.action=="new_user_success"):
                            pipe_sms_send_local.send(request_content)
                        elif(request_content.action=="prompt_overflow"):
                            pipe_sms_send_local.send(request_content)
                        elif(request_content.action=="prompt_rcvd"):
                            queue_size+=1
                            request_content.q_pos=queue_size
                            pipe_sms_send_local.send(request_content)
                            pipe_gpu_local.send(request_content)
                except Exception as e:
                    print(e)
                    continue
                    

            if(pipe_gpu_local.poll(0.1)):
                try:
                    request_content=pipe_gpu_local.recv()
                    queue_size -= 1
                    if(handle_gpu_request(request_content,client_db)):
                        pipe_sms_send_local.send(request_content)
                except Exception as e:
                    print(e)
                    continue
            if(time.time()>next_db_save):
                next_db_save=time.time()+db_save_delay
                client_db_save(client_db)
    except:
        print("saving client DB")
        client_db_save(client_db)

if __name__ == "__main__":
    with open('errlog.log', 'w') as stderr, redirect_stderr(stderr):
        main()
