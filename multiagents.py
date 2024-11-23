import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "Enter-OpenAI-Key"

# Data Models
class AgentPlan(BaseModel):
    """Represents a plan from an individual agent"""
    agent_id: str
    steps: List[Dict[str, Any]]
    confidence_score: Optional[float] = None

class PlanningState(BaseModel):
    """State for the multi-agent planning process"""
    user_input: str
    num_agents: int = 3
    agent_plans: List[AgentPlan] = []
    evaluated_plan: Optional[AgentPlan] = None
    final_plan: Optional[Dict[str, Any]] = None

class RequirementsCollector:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.2)
        self.template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a data collection agent. Your task is to gather detailed information from users about their task requirements. Ask relevant questions that would be required for planning according to their usecase:

            1. What task does the user want you to plan for
            2. Timeline and Deadlines
            3. Resources Required
            4. Constraints and Limitations
            5. Risk Factors to be considered 
            6. Feasibility 
            7. Any other details related to the context


            Continue asking questions until you have comprehensive information. Once all necessary details are collected, end with 'REQUIREMENTS GATHERED'.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

    async def collect_requirements(self) -> str:
        store = {}
        session_id = "requirements_session"
        
        chain = self.template | self.llm
        runnable = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=lambda: self._get_session_history(session_id, store),
            input_messages_key="input",
            history_messages_key="history"
        )

        collected_info = []
        print("🤖 Requirements Collector: Hello! I'll help gather requirements for your task. What would you like to plan?")
        
        while True:
            user_input = input()
            response = await runnable.ainvoke({"input": user_input})
            print(f"🤖 Requirements Collector: {response.content}")
            collected_info.append(response.content)
            
            if "REQUIREMENTS GATHERED" in response.content:
                return "\n".join(collected_info)

    def _get_session_history(self, session_id: str, store):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

class MultiAgentPlanner:
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=llm_model, temperature=0.3)
        self.requirements_collector = RequirementsCollector(llm_model)

    async def generate_prompt_for_agents(self, requirements: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are the Prompt Generation Agent. Create a comprehensive prompt for planning agents based on the collected requirements.
            Structure the prompt to help agents create detailed, actionable plans that address all aspects of the project.You are guiding me for planning trips.
            """),
            ("human", f"Requirements:\n{requirements}")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({})
        print(f"📝 Prompt Generator: Generated planning prompt based on requirements")
        return response.content

    async def generate_agent_plan(self, agent_id: str, prompt: str) -> AgentPlan:
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            You are Planning {agent_id}. Create a detailed, step-by-step plan based on the following prompt.
            Include specific actions, timelines, dependencies, and expected outcomes for each step.
            Structure your response as a series of clear, actionable steps.
            """),
            ("human", prompt)
        ])
        
        chain = agent_prompt | self.llm
        response = await chain.ainvoke({})
        
        plan = AgentPlan(
            agent_id=agent_id,
            steps=[{"step": i+1, "description": step.strip()} 
                   for i, step in enumerate(response.content.split("\n")) 
                   if step.strip()]
        )
        
        print(f"\n🤔 {agent_id} Proposal:")
        for step in plan.steps:
            print(f"   Step {step['step']}: {step['description']}")
        print()
        
        return plan

    async def evaluate_plans(self, plans: List[AgentPlan]) -> AgentPlan:
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are the Evaluation Agent. Review the provided plans and:
            1. Assign a confidence score (0-1) to each plan
            2. Select the best plan based on comprehensiveness, feasibility, and alignment with requirements
            3. Provide reasoning for your selection
            
            Format your response as:
            SELECTED_PLAN: [Agent ID]
            CONFIDENCE_SCORE: [Score]
            REASONING: [Your detailed explanation]
            """),
            ("human", "\n\n".join([
                f"Plan {plan.agent_id}:\n" + 
                "\n".join([f"Step {step['step']}: {step['description']}" 
                          for step in plan.steps])
                for plan in plans
            ]))
        ])
        
        chain = evaluation_prompt | self.llm
        response = await chain.ainvoke({})
        print(f"\n⚖️ Evaluation Agent:\n{response.content}\n")
        
        selected_plan = next(
            plan for plan in plans 
            if plan.agent_id in response.content.split("SELECTED_PLAN:")[1].split("\n")[0]
        )
        selected_plan.confidence_score = float(
            response.content.split("CONFIDENCE_SCORE:")[1].split("\n")[0]
        )
        
        return selected_plan

    async def create_planning_graph(self, final_plan: AgentPlan):
        graph = StateGraph(dict)
        
        for step in final_plan.steps:
            graph.add_node(str(step['step']), 
                          lambda state, step=step: self._execute_step(state, step))
        
        steps = final_plan.steps
        for i in range(len(steps) - 1):
            graph.add_edge(str(steps[i]['step']), str(steps[i+1]['step']))
        
        graph.add_edge(START, str(steps[0]['step']))
        graph.add_edge(str(steps[-1]['step']), END)
        
        return graph.compile()

    def _execute_step(self, state: Dict, step: Dict[str, Any]) -> Dict:
        logger.info(f"Executing step {step['step']}: {step['description']}")
        return state

    async def main(self):
        # Collect requirements
        requirements = await self.requirements_collector.collect_requirements()
        
        # Generate prompt for planning agents
        planning_prompt = await self.generate_prompt_for_agents(requirements)
        
        # Generate plans from multiple agents
        agent_plans = await asyncio.gather(*[
            self.generate_agent_plan(f"Agent_{i}", planning_prompt)
            for i in range(3)
        ])
        
        # Evaluate and select best plan
        best_plan = await self.evaluate_plans(agent_plans)
        
        print("\n🏆 === Selected Plan ===")
        print(f"Agent: {best_plan.agent_id}")
        print(f"Confidence Score: {best_plan.confidence_score}\n")
        for step in best_plan.steps:
            print(f"Step {step['step']}: {step['description']}")

async def run_planner():
    planner = MultiAgentPlanner()
    await planner.main()

if __name__ == "__main__":
    asyncio.run(run_planner())
