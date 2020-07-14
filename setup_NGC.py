from Azuremlcomputecluster import AzureMLComputeCluster
import ngccontent
from azureml.core import Workspace, Experiment, Datastore, Dataset, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import MpiConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.authentication import InteractiveLoginAuthentication

import os
from IPython.core.display import display, HTML
import argparse

def main():
    parser = argparse.ArgumentParser(description="NGC Set Up on AzureML Compute Cluster")
    parser.add_argument("--config_file", type=str, help="location of config file")
    args = parser.parse_args()
    config_file = args.config_file

    print(config_file)
    configdata = ngccontent.get_config(config_file)
    subscription_id = configdata["azureml_user"]["subscription_id"]
    resource_group = configdata["azureml_user"]["resource_group"]
    workspace_name = configdata["azureml_user"]["workspace_name"] 
    
    ws = Workspace(
        workspace_name=workspace_name
        , subscription_id=subscription_id
        , resource_group=resource_group
    )

    verify = f'''
    Subscription ID: {subscription_id}
    Resource Group: {resource_group}
    Workspace: {workspace_name}'''
    print(verify)
    
    ### vnet settings
    vnet_rg = ws.resource_group
    vnet_name = configdata["aml_compute"]["vnet_name"]
    subnet_name = configdata["aml_compute"]["subnet_name"]
    
    ### azure ml names
    ct_name  = configdata["aml_compute"]["ct_name"]
    exp_name = configdata["aml_compute"]["exp_name"]
    
    ### trust but verify
    verify = f'''
    vNET RG: {vnet_rg}
    vNET name: {vnet_name}
    vNET subnet name: {subnet_name}
    Compute target: {ct_name}
    Experiment name: {exp_name}'''
    print(verify)

    if configdata["aml_compute"]["vm_name"] in configdata["supported_vm_sizes"].keys():
        vm_name = configdata["aml_compute"]["vm_name"]
        gpus_per_node = configdata["supported_vm_sizes"][vm_name]
        
        print("Setting up compute target {ct_name} with vm_size: {vm_name} with {gpus_per_node} GPUs".format(ct_name=ct_name,vm_name=vm_name,gpus_per_node=gpus_per_node))
    
        if ct_name not in ws.compute_targets:
            config = AmlCompute.provisioning_configuration(
                vm_size=vm_name
                , min_nodes=configdata["aml_compute"]["min_nodes"]
                , max_nodes=configdata["aml_compute"]["max_nodes"]
                , vnet_resourcegroup_name=vnet_rg
                , vnet_name=vnet_name
                , subnet_name=subnet_name
                , idle_seconds_before_scaledown=configdata["aml_compute"]["idle_seconds_before_scaledown"]
                , remote_login_port_public_access='Enabled'
            )
            ct = ComputeTarget.create(ws, ct_name, config)
            ct.wait_for_completion(show_output=True)
        else:
            print("Loading Pre-existing Compute Target {ct_name}".format(ct_name=ct_name)) 
            ct = ws.compute_targets[ct_name]
    else:
        print("Unsupported vm_size {vm_size}".format(vm_size=vm_name))
        print("The specified vm size must be one of ...")
        for azure_gpu_vm_size in configdata["supported_vm_sizes"].keys():
            print("... " + azure_gpu_vm_size)
        raise Exception("{vm_size} does not support Pascal or above GPUs".format(vm_size=vm_name))

    environment_name=configdata["aml_compute"]["environment_name"]
    python_interpreter = configdata["aml_compute"]["python_interpreter"]
    conda_packages = configdata["aml_compute"]["conda_packages"]
    from azureml.core import ContainerRegistry
    
    if environment_name not in ws.environments:
        env = Environment(name=environment_name)
        env.docker.enabled = configdata["aml_compute"]["docker_enabled"]
        env.docker.base_image = None
        env.docker.base_dockerfile = "FROM {dockerfile}".format(dockerfile=configdata["ngc_content"]["base_dockerfile"])
        env.python.interpreter_path = python_interpreter
        env.python.user_managed_dependencies = True
        conda_dep = CondaDependencies()

        for conda_package in conda_packages:
            conda_dep.add_conda_package(conda_package)
    
        env.python.conda_dependencies = conda_dep
        env.register(workspace=ws)
        evn = env
    else:
        env = ws.environments[environment_name]
    
    amlcluster = AzureMLComputeCluster(
        workspace=ws
        , compute_target=ct
        , initial_node_count=1
        , experiment_name=configdata["aml_compute"]["exp_name"]
        , environment_definition=env
        , use_gpu=True
        , n_gpus_per_node=1
        , jupyter=True
        , jupyter_port=configdata["aml_compute"]["jupyter_port"]
        , dashboard_port=9001
        , scheduler_port=9002
        , scheduler_idle_timeout=1200
        , worker_death_timeout=30
        , additional_ports=[]
        , datastores=[]
        , telemetry_opt_out=True
        , asynchronous=False
    )

    print(amlcluster.jupyter_link)
    amlcluster.jupyter_link
    print('Exiting script')


if __name__ == '__main__':
    main()
