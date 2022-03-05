from navigation import Environment
from unityagents import UnityEnvironment
import torch


if __name__ == "__main__":
    env = Environment(UnityEnvironment(file_name="Banana_Linux/Banana.x86_64"))
    scores = env.run_model('checkpoint.pth')
    env.close()
