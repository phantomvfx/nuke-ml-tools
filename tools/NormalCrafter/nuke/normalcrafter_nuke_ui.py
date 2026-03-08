import nuke
import os
import sys
import subprocess
import tempfile
import threading

def run_generation(gizmo_node):
    # Check if there is an input connected
    if not gizmo_node.input(0):
        nuke.message("Please connect an input node to the NormalCrafter node.")
        return
        
    # Get parameters from the gizmo
    max_res = int(gizmo_node['max_res'].value())
    window_size = int(gizmo_node['window_size'].value())
    decode_chunk_size = int(gizmo_node['decode_chunk_size'].value())
    time_step_size = int(gizmo_node['time_step_size'].value())
    
    # Get frame range
    first_frame = int(nuke.root()['first_frame'].value())
    last_frame = int(nuke.root()['last_frame'].value())
    
    # Try to get frame range from input node if possible
    try:
        input_node = gizmo_node.input(0)
        first_frame = int(input_node['first'].value())
        last_frame = int(input_node['last'].value())
    except:
        pass
        
    # Create temp directory
    temp_dir = os.path.join(tempfile.gettempdir(), "NormalCrafter_Nuke")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    input_pattern = os.path.join(temp_dir, "input_%04d.exr").replace("\\", "/")
    output_pattern = os.path.join(temp_dir, "normals_%04d.exr").replace("\\", "/")
    
    # Render input to temp directory
    task = nuke.ProgressTask("NormalCrafter Generation")
    task.setMessage("Rendering input sequence...")
    
    # Temporarily create a Write node to render the frames
    write_node = nuke.nodes.Write(
        file=input_pattern,
        file_type="exr",
        datatype="16 bit half",
        compression="Zip (1 scanline)"
    )
    write_node.setInput(0, gizmo_node.input(0))
    
    # Execute render
    try:
        nuke.execute(write_node, first_frame, last_frame)
    except Exception as e:
        nuke.message(f"Failed to render input sequence: {e}")
        nuke.delete(write_node)
        return
        
    nuke.delete(write_node)
    
    # Set up python environment and command for subprocess
    script_dir = os.path.dirname(__file__)
    tool_root = os.path.dirname(script_dir)
    project_root = os.path.dirname(os.path.dirname(tool_root))
    
    # Path to the python executable in the nuke17_ml_env mamba environment
    python_exe = os.path.join(project_root, "nuke17_ml_env", "python.exe")
    wrapper_script = os.path.join(tool_root, "src", "run_normalcrafter_nuke.py")
    
    if not os.path.exists(python_exe):
        nuke.message(f"Could not find Python executable at: {python_exe}. Please ensure the nuke17_ml_env is installed.")
        return
        
    cmd = [
        python_exe,
        wrapper_script,
        "--input", input_pattern,
        "--output", output_pattern,
        "--max-res", str(max_res),
        "--window-size", str(window_size),
        "--decode-chunk-size", str(decode_chunk_size),
        "--time-step-size", str(time_step_size)
    ]
    
    task.setMessage("Executing Video Diffusion (NormalCrafter)... Check terminal for details.")
    task.setProgress(50)
    
    # Run subprocess
    try:
        print("Running command:", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        # We block nuke but ideally we would run this in a thread. For simplicity, we block.
        for line in process.stdout:
            print("NormalCrafter:", line.strip())
            if task.isCancelled():
                process.terminate()
                nuke.message("Generation cancelled by user.")
                return
                
        process.wait()
        
        if process.returncode != 0:
            nuke.message(f"NormalCrafter process failed with return code {process.returncode}. See Nuke console for details.")
            return
            
    except Exception as e:
        nuke.message(f"Error running NormalCrafter subprocess: {e}")
        return
        
    task.setProgress(100)
    del task
    
    # Create Read node for output
    read_node = nuke.nodes.Read(
        file=output_pattern,
        first=first_frame,
        last=last_frame
    )
    
    # Place it below the current gizmo
    read_node['xpos'].setValue(gizmo_node['xpos'].value())
    read_node['ypos'].setValue(gizmo_node['ypos'].value() + 60)
    
    # Select the new node
    for n in nuke.allNodes():
        n.setSelected(False)
    read_node.setSelected(True)
    
    print("NormalCrafter generation completed successfully!")
