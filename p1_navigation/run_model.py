from unityagents import UnityEnvironment

from navigation import Environment

if __name__ == "__main__":
    env = Environment(UnityEnvironment(file_name="Banana_Linux/Banana.x86_64"), 12)
    scores = env.run_model('checkpoint.pth')
    env.close()
