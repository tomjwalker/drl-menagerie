from tac.env.openai import OpenAIEnv


def make_env(spec):
    env = OpenAIEnv(spec)
    return env
