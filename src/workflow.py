import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name], check=True)

def main():
    run_script("src/qiskit_noise.py")
    run_script("src/gnn_regressor.py")

if __name__ == "__main__":
    main()