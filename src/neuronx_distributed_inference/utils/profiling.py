import os
import json
import subprocess
import traceback

from neuronx_distributed_inference.utils.constants import (
    VISION_ENCODER_MODEL,
)


PROFILE_BASE_DIR = "./profile/"


def run_shell_command(cmd_str):
    print("Running command: " + cmd_str)
    try:
        process = subprocess.run(cmd_str, capture_output=True, shell=True)
        process.check_returncode()
    except Exception as exception:
        print(f"Got an exception when running command {cmd_str}")
        stdout = process.stdout.decode('utf-8')
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {process.stderr.decode('utf-8')}")
        print(traceback.format_exc())
        if "execution completed with numerical error (NaN)" in stdout:
            # profile.ntff is generated successfully
            print("Given that the ntff is generated successfully, ignoring the NaN numerical error from neuron-profiling")
        else:
            raise exception
    return process.stdout.decode("utf-8")


def run_profiler_on_neff(neff_path, output_ntff_folder, world_size):
    os.makedirs(output_ntff_folder, exist_ok=True)
    ntff_prefix = output_ntff_folder + "profile"
    capture_command = ["/opt/aws/neuron/bin/neuron-profile", "capture",
                       "-n", neff_path, "-s", ntff_prefix + ".ntff",
                       # Run world_size workers for collectives
                       "--collectives-workers-per-node", str(world_size),
                       # Generate ntff only for worker 0
                       "--collectives-profile-id 0",
                       # Run two executions
                       "--num-exec 2",
                       # Profile the second execution (first is warmup)
                       "--profile-nth-exec 2",
                       # Ignore errors
                       "--ignore-exec-errors",
                       ]
    capture_command = " ".join(capture_command)
    run_shell_command(capture_command)

    # Use the data from the second execution (to exclude one-time initialization overheads)
    ntff_path = ntff_prefix + "_rank_0_exec_2.ntff"
    view_command = ["/opt/aws/neuron/bin/neuron-profile", "view",
                    "-n", neff_path, "-s", ntff_path,
                    "--output-format", "summary-json",
                    "--ignore-nc-buf-usage"]
    view_command = " ".join(view_command)
    output = run_shell_command(view_command)
    metrics = list(json.loads(output).values())[0]
    print("Profiling metrics: " + str(metrics))

    return metrics, neff_path, ntff_path


def get_largest_bucket_dir(submodel_compiler_work_dir):
    '''
    return path to the largest bucket for profiling
    '''
    # Get all subdirectories
    bucket_dirs = [d for d in os.listdir(submodel_compiler_work_dir)
                   if os.path.isdir(os.path.join(submodel_compiler_work_dir, d))]

    # Sort them in default ascending order
    bucket_dirs.sort()

    if not bucket_dirs:
        return None

    # Get the last one and return its full path
    last_bucket_dir = bucket_dirs[-1]
    return os.path.join(submodel_compiler_work_dir, last_bucket_dir)


def get_neff_path_and_output_ntff_folder(submodel_name, bucket_index, compiler_workdir=None):
    base_compile_workdir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
    perf_test_compile_workdir = os.path.join(base_compile_workdir, "perf_test")
    if os.path.exists(perf_test_compile_workdir):
        # Use the NEFF from the perf test flow if present.
        base_compile_workdir = perf_test_compile_workdir
    # Use specified compiler_workdir if given
    if compiler_workdir is not None:
        base_compile_workdir = compiler_workdir

    # Temporary fix to map text model subdir correctly for Llama4/Pixtral.
    # Llama4/Pixtral currently only profile the text submodels (not vision).
    # TODO: Add general solution to NeuronApplicationBase that maps submodel name to compiler workdir.
    text_model_compile_workdir = os.path.join(base_compile_workdir, "text_model")
    vision_model_compile_workdir = os.path.join(base_compile_workdir, "vision_model")
    if submodel_name == VISION_ENCODER_MODEL:
        base_compile_workdir = vision_model_compile_workdir
    elif os.path.exists(text_model_compile_workdir):
        base_compile_workdir = text_model_compile_workdir

    submodel_compiler_work_dir = os.path.join(base_compile_workdir, submodel_name)
    if bucket_index is not None:
        non_bucketed_neff_path = os.path.join(submodel_compiler_work_dir, "_tp0/graph.neff")
        bucketed_neff_path = os.path.join(submodel_compiler_work_dir, f"_tp0_bk{bucket_index}/graph.neff")
        if (bucket_index == 0) and os.path.exists(non_bucketed_neff_path):
            neff_path = non_bucketed_neff_path
        else:
            neff_path = bucketed_neff_path
    else:
        # if bucket_index is not specified, we profile the largest bucket
        largest_bucket_dir = get_largest_bucket_dir(submodel_compiler_work_dir)
        neff_path = os.path.join(largest_bucket_dir, "graph.neff")

    assert os.path.exists(neff_path), f"neff_path: {neff_path}"

    output_ntff_folder = PROFILE_BASE_DIR + submodel_name + "/"
    return neff_path, output_ntff_folder
