import subprocess

def run_script(script_name):
    subprocess.run(["python", script_name], check=True)

def main():
    run_script("src/training_data_generation.py")
    run_script("src/test_data_generation.py")
    run_script("src/train.py")

if __name__ == "__main__":
    main()