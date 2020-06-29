# Imports
import os
import sys
import uuid
import time
import socket
import argparse
import threading
import subprocess
import logging
####---from mpi4py import MPI

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def flush(proc, proc_log):
    while True:
        proc_out = proc.stdout.readline()
        if proc_out == "" and proc.poll() is not None:
            proc_log.close()
            break
        elif proc_out:
            sys.stdout.write(proc_out)
            proc_log.write(proc_out)
            proc_log.flush()

def evaluate_cmd(cmd, logFileName):
    cmd_log = open(logFileName, "a")
    cmd_proc = subprocess.Popen(
        cmd.split(),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    cmd_flush = threading.Thread(target=flush, args=(cmd_proc, cmd_log))
    cmd_flush.start()
    flush(cmd_proc, cmd_log)
    return cmd_proc


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    install("azureml-sdk")
    install("notebook")
    from azureml.core import Run
    from notebook.notebookapp import list_running_servers

# ###---    comm = MPI.COMM_WORLD
# ###---    rank = comm.Get_rank()

    ### PARSE ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--jupyter", default=False)
    parser.add_argument("--jupyter_token", default=uuid.uuid1().hex)
    parser.add_argument("--jupyter_port", default=8888)
    parser.add_argument("--dashboard_port", default=8787)
    parser.add_argument("--scheduler_port", default=8786)
    parser.add_argument("--scheduler_idle_timeout", default=1200)  # 20 mins
    parser.add_argument("--worker_death_timeout", default=300)  # 5 mins
    parser.add_argument("--use_gpu", default=False)
    parser.add_argument("--n_gpus_per_node", default=0)

    args, unparsed = parser.parse_known_args()

    ### CONFIGURE GPU RUN
    GPU_run = args.use_gpu

    if GPU_run:
        n_gpus_per_node = eval(args.n_gpus_per_node)

    ip = socket.gethostbyname(socket.gethostname())

    ### SETUP THE HEADNODE
####---    if rank == 0:
    data = {
        "scheduler": ip + ":" + str(args.scheduler_port),
        "scheduler_idle_timeout": str(args.scheduler_idle_timeout),
        "worker_death_timeout": str(args.worker_death_timeout),
        "dashboard": ip + ":" + str(args.dashboard_port),
        "jupyter": ip + ":" + str(args.jupyter_port),
        "token": args.jupyter_token,
    }
####---     else:
####---         data = None

    ### DISTRIBUTE TO CLUSTER
####---    data = comm.bcast(data, root=0)
    scheduler = data["scheduler"]
    scheduler_idle_timeout = data["scheduler_idle_timeout"]
    worker_death_timeout = data["worker_death_timeout"]
    dashboard = data["dashboard"]
    jupyter = data["jupyter"]
    token = data["token"]

    logger.debug("- scheduler is ", scheduler)
    logger.debug("- scheduler timeout is ", scheduler_idle_timeout)
    logger.debug("- worker timeout is ", worker_death_timeout)
    logger.debug("- dashboard is ", dashboard)
    logger.debug("- args: ", args)
    logger.debug("- unparsed: ", unparsed)
####---    logger.debug("- my rank is ", rank)
    logger.debug("- my ip is ", ip)

    ####---# if rank == 0:
    running_jupyter_servers = list(list_running_servers())
    logger.debug("- running jupyter servers", running_jupyter_servers)

    ### if any jupyter processes running
    ### KILL'EM ALL!!!
    if len(running_jupyter_servers) > 0:
        for server in running_jupyter_servers:
            os.system(f'kill {server["pid"]}')

    ### RECORD LOGS
    run = Run.get_context()

    run.log("headnode", ip)
    run.log("scheduler", scheduler)
    run.log("scheduler_idle_timeout", scheduler_idle_timeout)
    run.log("worker_death_timeout", worker_death_timeout)
    run.log("dashboard", dashboard)
    run.log("jupyter", jupyter)
    run.log("token", token)

    workspace_name = run.experiment.workspace.name.lower()
    run_id = run.get_details()["runId"]

    mount_point = f"/mnt/batch/tasks/shared/LS_root/jobs/{workspace_name}/azureml/{run_id.lower()}/mounts/"

    if args.jupyter:
        cmd = (
            f" jupyter lab --ip 0.0.0.0 --port {args.jupyter_port}"
            f" --NotebookApp.token={token}"
        )
        cmd += f" --notebook-dir={mount_point}"
        cmd += f" --allow-root --no-browser"

        jupyter_log = open("jupyter_log.txt", "a")
        jupyter_proc = subprocess.Popen(
            cmd.split(),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        jupyter_flush = threading.Thread(target=flush, args=(jupyter_proc, jupyter_log))
        jupyter_flush.start()

        while not list(list_running_servers()):
            time.sleep(5)

        jupyter_servers = list(list_running_servers())
        assert len(jupyter_servers) == 1, "more than one jupyter server is running"
        
        flush(jupyter_proc, jupyter_log)
        
    
    extraCMDS = {f"wget https://api.ngc.nvidia.com/v2/resources/nvidia/med/getting_started/versions/1/zip":"unload_log.txt",f"mv zip zip.zip":"mv_log.txt",f"unzip zip.zip":"unzip_log.txt"}
    extraCMDS_procs = {}
    
    for key in extraCMDS:
        extraCMDS_procs[key] = evaluate_cmd(key,extraCMDS[key])

    if jupyter_proc:
        jupyter_proc.kill()

    for key in extraCMDS_procs:
        if extraCMDS_procs[key]:
            extraCMDS_procs[key].kill()
    
    run.complete()
    run.cancel()
