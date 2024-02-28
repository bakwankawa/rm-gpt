from copy import deepcopy
from typing import Any, Dict, List, Union

from langchain import LLMChain
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field

from chains import SalesConversationChain, StageAnalyzerChain
from parser_1 import SalesConvoOutputParser
from templates import CustomPromptTemplateForTools
from tools import get_tools, get_retrieval
from prompts import finalPrompt
import streamlit as st

from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

# import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

class SalesGPT(Chain, BaseModel):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    current_conversation_stage: str = "1"
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)

    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    use_tools: bool = False

    # st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False, collapse_completed_thoughts=False)

    conversation_stage_dict: Dict = {
        "1": "Introduction: Start the conversation by introducing yourself and your company. Be polite and respectful while keeping the tone of the conversation professional. Your greeting should be welcoming. Always clarify in your greeting the reason why you are contacting the prospect.",
        "2": "Value proposition: Briefly explain how your product/service can benefit the prospect. Focus on the unique selling points and value proposition of your product/service that sets it apart from competitors.",
        "3": "Solution presentation: Based on the prospect's needs, present your product/service as the solution that can address their pain points.",
    }

    salesperson_name="Sabrina"
    salesperson_role="Relationship Manager Representative"
    company_name="Bank Rakyat Indonesia"
    company_business="Bank Rakyat Indonesia is one of the largest banks in Indonesia. It specialises in small scale and microfinance style borrowing from and lending to its approximately 30 million retail clients through its over 8,600 branches, units and rural service posts."
    company_values="Carry out the best banking activities by prioritizing services to the micro, small and medium segments to support the improvement of the people's economy"
    conversation_purpose="offer BRI credit card products."
    conversation_type="call"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []

    def determine_conversation_stage(self):
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history='"\n"'.join(self.conversation_history),
            current_conversation_stage=self.current_conversation_stage,
        )

        self.current_conversation_stage = self.retrieve_conversation_stage(
            conversation_stage_id
        )

        print(f"Conversation Stage: {self.current_conversation_stage}")
        return self.current_conversation_stage

    def human_step(self, human_input):
        # process human input
        human_input = "User: " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)

    def step(self):
        return self._call(inputs={})

    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""

        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.sales_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
                callbacks=[StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)],
                # callbacks=[FinalStreamingStdOutCallbackHandler()]
            )

        else:
            ai_message = self.sales_conversation_utterance_chain.run(
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                company_values=self.company_values,
                conversation_purpose=self.conversation_purpose,
                conversation_history="\n".join(self.conversation_history),
                conversation_stage=self.current_conversation_stage,
                conversation_type=self.conversation_type,
                # callbacks=[StreamlitCallbackHandler(st.container())]
            )

        # Add agent's response to conversation history
        print(f"{self.salesperson_name}: ", ai_message.rstrip("<END_OF_TURN>"))
        agent_name = self.salesperson_name
        ai_message = agent_name + ": " + ai_message
        if "<END_OF_TURN>" not in ai_message:
            ai_message += " <END_OF_TURN>"
        self.conversation_history.append(ai_message)
        # print(ai_message)

        return ai_message

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        sales_conversation_utterance_chain = SalesConversationChain.from_llm(
            llm, verbose=verbose
        )

        if "use_tools" in kwargs.keys() and kwargs["use_tools"] is False:
            sales_agent_executor = None

        else:
            tools = get_tools(get_retrieval())
            # print(tools)

            # final_prompt = prefix
            # final_prompt += questionSemantic(humanInput)
            # final_prompt += suffix

            # print(final_prompt)

            prompt = CustomPromptTemplateForTools(
                template=finalPrompt(),
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "salesperson_name",
                    "salesperson_role",
                    "company_name",
                    "company_business",
                    "company_values",
                    "conversation_purpose",
                    "conversation_type",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose) # ini yang harus di dynamic in!!!

            tool_names = [tool.name for tool in tools]
            # print(tool_names)

            # WARNING: this output parser is NOT reliable yet
            ## It makes assumptions about output from LLM which can break and throw an error
            output_parser = SalesConvoOutputParser(ai_prefix=kwargs["salesperson_name"])

            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                verbose=verbose,
                return_intermediate_steps=False,
            )

            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools, tools=tools, verbose=False, max_iteratrions=3,
            )

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            verbose=verbose,
            **kwargs,
        )