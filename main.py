import os, dotenv
from agents import SalesGPT
from langchain.chat_models import ChatOpenAI
from time import sleep
import streamlit as st
from streamlit_chat import message
from langchain.callbacks import StreamlitCallbackHandler
import time
from langchain.callbacks import get_openai_callback
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + "/"
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")
        
chat_box = st.empty()
stream_handler = StreamHandler(chat_box, display_method='write')

# Conversation stages - can be modified
conversation_stages = {
    "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
    "2": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
    "3": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
}

# Agent characteristics - can be modified
config = dict(
    salesperson_name="Sabrina",
    salesperson_role="Sales Representative",
    company_name="Bank Rakyat Indonesia",
    company_business="Bank Rakyat Indonesia is one of the largest banks in Indonesia. It specialises in small scale and microfinance style borrowing from and lending to its approximately 30 million retail clients through its over 8,600 branches, units and rural service posts.",
    company_values="Carry out the best banking activities by prioritizing services to the micro, small and medium segments to support the improvement of the people's economy",
    conversation_purpose="offering them BRI credit card.",
    conversation_history=[],
    conversation_type="call",
    conversation_stage=conversation_stages.get(
        "1",
        "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional.",
    ),
    use_tools=True,
)


if 'generated' not in st.session_state:
    st.session_state.generated = []

if 'past' not in st.session_state:
    st.session_state.past = []


# Initialize session state
if 'sales_agent' not in st.session_state:
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo-16k', streaming=True)
    sales_agent = SalesGPT.from_llm(llm, verbose=True, **config)
    sales_agent.seed_agent()
    sales_agent.determine_conversation_stage()
    initial_chat = sales_agent.step()
    # print(len(st.session_state['generated']))
    st.session_state.past.append("...")
    st.session_state.generated.append(initial_chat)
    st.session_state.sales_agent_initialized = True
    st.session_state.sales_agent = sales_agent

# Function to display animated chat messages    
def animate_chat_message(message, is_user=False):
    if is_user:
        st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message chatbot-message">{message}</div>', unsafe_allow_html=True)

def main():
    st.title('🦜🔗 RM-GPT')

    # Get user input
    user_input = st.text_input("You:", key="input")

    if user_input:
        sales_agent = st.session_state.sales_agent
        sales_agent.human_step(user_input)
        sales_agent.determine_conversation_stage()
        output = sales_agent.step()
        # Remove "<END_OF_TURN>" from the output
        output = output.replace("<END_OF_TURN>", "")
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state.generated:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i in range(len(st.session_state.generated)-1, -1, -1):
            animate_chat_message(st.session_state.generated[i])
            time.sleep(0.5)  # Adjust the delay time as needed
            animate_chat_message(st.session_state.past[i], is_user=True)
            time.sleep(0.5)  # Adjust the delay time as needed
        st.markdown('</div>', unsafe_allow_html=True)

# Inject custom CSS styles
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
image_url = "https://bri.co.id/documents/20123/57149/Sabrina+Pose1+%282%29.png/71a72847-30b4-d618-5842-45dfcaf0a631?t=1657158831419"
import streamlit as st

css = """
<style>
    .chat-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        align-items: center;
        height: calc(100% - 80px); /* Subtract input box height */
        padding: 10px;
        background-color: #ffffff;
        border-radius: 8px;
        overflow-y: auto;
    }

    .chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 10px;
        background-color: #f0f0f0;
        border-top: 1px solid #ccc;
        z-index: 1; /* Ensure it's above the chat history */
    }

    .chat-message {
        margin: 8px 0;
        padding: 8px 12px;
        border-radius: 8px;
        max-width: 80%;
    }
    .user-message {
        background-color: #fc7411;
        color: #333;
        margin-right: auto;
    }
    .chatbot-message {
        background-color: #014a94;
        color: white;
        margin-left: auto;
    }
    .cropped-image {
        display: block;
        margin: 0 auto; /* Center-align the image */
        max-height: 400px; /* Adjust the height as needed */
        overflow: hidden;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)
st.markdown('<div class="chat-container"><div class="cropped-image"><img src="' + image_url + '" alt="Cropped Image"></div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
