import argparse
import logging
import os
import shlex
import subprocess
import sys
import uuid
from typing import List


def get_command_output(command: List[str], exit_on_err=True):
    output = None
    try:
        proc = subprocess.run(
            command,
            shell=False,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = proc.stdout.decode("utf-8")
    except Exception as exc:
        logging.error(f"ERROR: command {' '.join(command)} failed with error: {exc}")
        if exit_on_err:
            exit(1)
    return output


def build_mpirun_command(parsed_args):
    mpirun_path = get_command_output(["which", "mpirun"]).rstrip()
    mpirun_dir = os.path.dirname(os.path.dirname(mpirun_path))

    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.path.join(mpirun_dir, 'lib')}:"
        f"{os.path.join(mpirun_dir, 'lib64')}:/opt/aws/neuron/lib:/opt/amazon/efa/lib"
        f"{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    if "NEURON_RT_ROOT_COMM_ID" not in os.environ:
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{parsed_args.master_addr}:45654"

    # > disable the usage of OFI (libfabric) component in the mtl framework
    # > only use tcp in the mca framework (self is always required)
    # > do not allow MPI to bind processes/threads to vCPU cores
    cmd = f"{mpirun_path} --mca mtl ^ofi --mca btl tcp,self --bind-to none "
    cmd += f"-N {parsed_args.nproc_per_node} -n {len(parsed_args.hosts)} "
    cmd += f"--host {','.join(parsed_args.hosts)} "
    for env_entry in ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
        if env_entry in os.environ and os.environ[env_entry]:
            cmd += f"-x {env_entry}={os.environ[env_entry]} "
    for env_entry, value in os.environ.items():
        if (
            env_entry.startswith("NEURON_")
            or env_entry.startswith("NCCL_")
            or env_entry.startswith("CCOM_")
            or env_entry.startswith("FI_")
            or env_entry.startswith("OFI_NCCL")
            or env_entry.startswith("XLA_")
        ) and value != "":
            cmd += f"-x {env_entry}={value} "
    cmd += "-x FI_PROVIDER=efa -x FI_EFA_USE_DEVICE_RDMA=1 -x FI_EFA_FORK_SAFE=1 "
    cmd += f" -x MASTER_ADDR={parsed_args.master_addr} -x MASTER_PORT={parsed_args.master_port} "

    if parsed_args.torchrun:
        cmd += build_torchrun_command(parsed_args)

    cmd += f"{' '.join(parsed_args.command)}"
    return cmd


def build_torchrun_command(parsed_args):
    cmd = (
        f"torchrun --nnodes {len(parsed_args.hosts)} --nproc-per-node {parsed_args.nproc_per_node} "
    )
    cmd += "--rdzv-backend c10d "
    cmd += f"--rdzv-id nxdi-{str(uuid.uuid4()).split('-')[0]} "
    cmd += f"--rdzv-endpoint {parsed_args.master_addr}:{parsed_args.master_port} "
    if not parsed_args.command[0].endswith(".py"):
        cmd += "--no-python "
    return cmd


def parse_arguments(argv: List[str]):
    parser = argparse.ArgumentParser(
        usage="nxdi_distributed_launcher <launcher_args> -- <command>",
        description="Distributed launcher utility for NxDI",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="mpi",
        const="all",
        nargs="?",
        choices=("mpi",),
        help="Distributed launcher backend to use. Currently only supports mpi",
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        action="store",
        help="Hosts on which to run execution",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Total number processes to launch on each node. Should be equal to 1 for NxDI",
    )
    parser.add_argument(
        "--torchrun",
        action="store_true",
        help="Prepend torchrun arguments",
    )
    parser.add_argument(
        "--master-addr",
        type=int,
        default=None,
        help="Address of the master node. It should be either the IP address or the hostname of rank 0.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=4567,
        help="Total number processes to launch on each node. Should be equal to 1 for NxDI",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to be executed",
    )

    parsed_args = parser.parse_args(argv)

    if parsed_args.hosts is None:
        parsed_args.hosts = ["127.0.0.1"]

    if parsed_args.master_addr is None:
        parsed_args.master_addr = parsed_args.hosts[0]

    assert (
        len(parsed_args.command) > 1 and parsed_args.command[0] == "--"
    ), "Please ensure the command to execute is separated by -- (double hyphen)"
    parsed_args.command = [shlex.quote(arg) for arg in parsed_args.command[1:]]

    return parsed_args


def main():
    parsed_args = parse_arguments(sys.argv[1:])
    cmd = build_mpirun_command(parsed_args)
    print(f"executing command: '{cmd}'", flush=True)
    os.execv("/bin/sh", ["/bin/sh", "-c", cmd])


if __name__ == "__main__":
    main()
