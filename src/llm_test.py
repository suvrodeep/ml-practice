from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.callbacks.manager import CallbackManager
from torch import cuda
import textwrap


def get_llm(model_path: str = None):
    if model_path is None:
        model_path = "../models/quantized_llms/mistral-7b-instruct-v0.2.Q5_K_M.gguf"

    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    if cuda.is_available():
        n_gpu_layers = 40  # Number of layers to offload to GPU. Depends on model and GPU VRAM pool.
        n_batch = 512  # Should be between 1 and n_ctx. Depends on VRAM in GPU.
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            # callback_manager=callback_manager,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            max_tokens=1024,
            n_ctx=1024,
            top_p=0.7,
            repeat_penalty=1.1,
            verbose=True  # Verbose is required to pass to the callback manager
        )
    else:
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            # callback_manager=callback_manager,
            max_tokens=1024,
            n_ctx=1024,
            top_p=0.7,
            repeat_penalty=1.1,
            verbose=True  # Verbose is required to pass to the callback manager
        )

    return llm


LLM = get_llm()


def build_chain(llm=None):
    if llm is None:
        llm = LLM

    prompt_template = """<s>[INST] 
        {query}  
        [/INST]"""

    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    return llm_chain


def main():
    llm_chain = build_chain()
    while True:
        query = input("\nPlease enter query: ")
        if query is None or query == "":
            print("\nNo input provided. Exiting...")
            exit(0)
        else:
            response = llm_chain.invoke(query)
            print(textwrap.fill(response["text"], width=100))


if __name__ == "__main__":
    main()













