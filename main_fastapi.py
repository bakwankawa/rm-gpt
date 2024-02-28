import os, dotenv
import logging
from typing import Optional
from agents import SalesGPT

from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import (
    QuestionGenCallbackHandler,
    StreamingLLMCallbackHandler,
    CustomStreamingStdOutCallbackHandler,
)
from schemas import ChatResponse
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Conversation stages - can be modified
conversation_stages = {
    "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
    "2": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
    "3": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
}

# Agent characteristics - can be modified
config = dict(
    salesperson_name="KawaKibi",
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

@app.on_event("startup")
async def startup_event():
    logging.info("loading tools")

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)

    while True:
        try:
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            selected_examples = example_selector.select_examples({"question": question})
            agent_prompt_template = ""

            for i, example in enumerate(selected_examples):
                agent_prompt_template += "[Example #{}]\n".format(i + 1)
                agent_prompt_template += "Question: {}".format(example["question"])
                agent_prompt_template += "{}\n".format(example["answer"])

            agent_prompt_template = (
                prefix_agent_template + agent_prompt_template + suffix_agent_template
                # suffix_agent_template
            )

            agent_executor = create_agent(
                agent_prompt_template, question_handler, stream_handler
            )
            result = await agent_executor.acall(
                {"input": question}, return_only_outputs=True
            )

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
