import asyncio
import logging
import os
import json
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI

# Set OpenAI API Key - IMPORTANT: Replace with your actual key or use environment variable
os.environ["OPENAI_API_KEY"] = "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"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
class AgentPlan(BaseModel):
    """Represents a plan from an individual agent"""
    agent_id: str
    steps: List[Dict[str, Any]]

class PlanningState(BaseModel):
    """State for the multi-agent planning process"""
    user_input: str
    num_agents: int = 3
    agent_plans: List[AgentPlan] = []
    evaluated_plan: Optional[AgentPlan] = None
    final_plan: Optional[Dict[str, Any]] = None

class MultiAgentPlanner:
    def __init__(self, 
                 llm_model: str = "gpt-4o-mini", 
                 max_workers: int = 3):
        """
        Initialize MultiAgentPlanner with configurable LLM and concurrency
        
        Args:
            llm_model (str): Language model to use for planning
            max_workers (int): Maximum number of concurrent agent threads
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=0.7)
        self.thread_executor = ThreadPoolExecutor(max_workers=max_workers)

    
    async def generate_prompt_for_agents(self, user_input: str) -> str:
        """
        Generate a comprehensive prompt for planning agents
        
        Args:
            user_input (str): Original user input
        
        Returns:
            str: Structured prompt for agents
        """
        prompt_generation_agent = (
            f"You are the Prompt Generation Agent. Analyze the user input below and generate a detailed and comprehensive "
            f"prompt for planning agents that will help them break down the task into actionable steps.\n\n"
            f"User Input: {user_input}\n\n"
            f"Consider the following in the generated prompt:\n"
            f"1. Key objectives of the task\n"
            f"2. Potential challenges and how to overcome them\n"
            f"3. Resources that might be needed\n"
            f"4. Actionable and sequential steps to achieve the task\n"
            f"5. Optimization and efficiency considerations\n"
            f"6. Any risks or dependencies that should be considered"
        )
        
        # Invoke LLM to generate the prompt
        prompt_response = await self.llm.ainvoke(prompt_generation_agent)
        
        # Print the generated prompt
        print(f"Prompt Generation Agent Output: {prompt_response.content}")
        
        return prompt_response.content
    
    def _parse_plan_response(self, response) -> List[Dict[str, Any]]:
        """
        Parse LLM response directly, attempting to extract structured plan steps
        
        Args:
            response: LLM plan response
        
        Returns:
            List of step dictionaries extracted from the LLM response
        """
        try:
            response_content = response.content if hasattr( response, 'content') else str(response)
            
            # Try to extract JSON if the response contains a JSON-like structure
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(0))
                    if isinstance(parsed_json, list) or isinstance(parsed_json, dict):
                        return [parsed_json] if isinstance(parsed_json, dict) else parsed_json
                except json.JSONDecodeError:
                    pass

            steps = []
            raw_steps = re.split(r'\n(?=\d+\.|\•|\-)', response_content)
            
            for idx, step_text in enumerate(raw_steps, 1):
                cleaned_step = step_text.strip()
                if cleaned_step:
                    steps.append({
                        "step": idx,
                        "action": cleaned_step.split('.')[0] if '.' in cleaned_step else cleaned_step,
                        "description": cleaned_step
                    })
            
            return steps
        except Exception as e:
            logger.error(f"Error parsing plan response: {e}")
            return [{"step": 1, "action": "Generated Plan", "description": response_content}]

    async def generate_agent_plan(self, agent_id: str, prompt: str) -> AgentPlan:
        """
        Generate a plan for a specific agent
        
        Args:
            agent_id (str): Unique identifier for the agent
            prompt (str): Comprehensive planning prompt
        
        Returns:
            AgentPlan: Detailed plan with confidence score
        """
        try:
            agent_specific_prompt = (
                f"You are Agent {agent_id}. Develop a comprehensive, detailed plan "
                f"for the following task: {prompt}\n\n"
                "Provide your plan with clear, actionable steps. "
                "Include specific details, potential challenges, "
                "and the reasoning behind each step."
            )
            
            plan_response = await self.llm.ainvoke(agent_specific_prompt)
            
            # Print agent plan response
            print(f"Agent {agent_id} Plan Output: {plan_response.content}")
            
            steps = self._parse_plan_response(plan_response)
            
            return AgentPlan(
                agent_id=agent_id,
                steps=steps
            )
        except Exception as e:
            logger.error(f"Error in agent {agent_id} planning: {e}")
            return AgentPlan(agent_id=agent_id, steps=[])

    async def calculate_confidence_score(self, plan: AgentPlan) -> float:
        """
        Calculate confidence score based on completeness, reliability, and value of the plan
        
        Args:
            plan (AgentPlan): The generated plan
        
        Returns:
            float: Confidence score based on evaluation of the plan
        """
        evaluation_prompt = (
            f"Please evaluate the following plan based on its completeness, reliability, and the value of the information it holds. "
            "Provide a confidence score from 0 to 1, where 0 is the lowest and 1 is the highest. "
            "Also, provide reasoning for your score.\n\n"
            f"Plan {plan.agent_id}:\n" + "\n".join([f"Step {step['step']}: {step['action']} - {step['description']}" for step in plan.steps])
        )
        
        eval_response = await self.llm.ainvoke(evaluation_prompt)
        
        print(f"Evaluation Response: {eval_response.content}")
        
        # Extract the confidence score and reason from the model's response
        match = re.search(r"\*\*Confidence Score:\*\*\s*(\d+\.\d+)", eval_response.content)
        if match:
            confidence_score = float(match.group(1))
            reason = re.search(r"Reasoning: (.*)", eval_response.content)
            reason_text = reason.group(1) if reason else "No reasoning provided."
            print("------////////--------", reason_text)
            return confidence_score, reason_text
        
        return 0.5, "No clear evaluation provided."

    async def evaluate_plans(self, plans: List[AgentPlan]) -> AgentPlan:
        """
        Evaluate and select the best plan using LLM
        
        Args:
            plans (List[AgentPlan]): Generated agent plans
        
        Returns:
            AgentPlan: Selected best plan
        """
        plans_text = "\n\n".join([f"Plan {plan.agent_id}:\n" + "\n".join([f"Step {step['step']}: {step['action']} - {step['description']}" for step in plan.steps])
            for plan in plans
        ])
        
        evaluation_prompt = (
            "Here are the plans for the following task. Please evaluate each plan based on completeness, reliability, and value, "
            "and give a confidence score from 0 to 1. Also, provide reasoning for each score:\n\n"
            f"{plans_text}\n\n"
            "After scoring, please choose the best plan based on the highest confidence score, and explain why it is the best."
        )
        
        eval_response = await self.llm.ainvoke(evaluation_prompt)
        
        print(f"Evaluation Response: {eval_response.content}")
        
        best_plan_id = re.search(r"Plan (\w+)", eval_response.content)
        if best_plan_id:
            best_plan_id = best_plan_id.group(1)
            best_plan = next((plan for plan in plans if plan.agent_id == best_plan_id), None)
            
            # Print the selected plan without the confidence score
            print("\n--- Selected Plan ---")
            for step in best_plan.steps:
                print(f"Step {step['step']}: {step['action']}")
                print(f"Description: {step['description']}\n")
            
            return best_plan
        
        return None
    
    async def create_planning_graph(self, final_plan: AgentPlan):
        """
        Create a LangGraph StateGraph from the final plan
        
        Args:
            final_plan (AgentPlan): Selected plan to convert to graph
        
        Returns:
            Compiled LangGraph
        """
        graph = StateGraph(dict)
        
        # Add nodes for each step
        for step in final_plan.steps:
            graph.add_node(str(step['step']), 
                           lambda state, step=step: self._execute_step(state, step))
        
        # Add edges between steps (simple linear for now)
        steps = final_plan.steps
        for i in range(len(steps) - 1):
            graph.add_edge(str(steps[i]['step']), str(steps[i+1]['step']))
        
        graph.add_edge(START, str(steps[0]['step']))
        graph.add_edge(str(steps[-1]['step']), END)
        
        return graph.compile()
    
    def _execute_step(self, state: Dict, step: Dict[str, Any]) -> Dict:
        """
        Execute an individual step in the plan
        
        Args:
            state (Dict): Current graph state
            step (Dict): Step to execute
        
        Returns:
            Updated state
        """
        logger.info(f"Executing step: {step.get('action', 'Unnamed Step')}")
        logger.info(f"Step Description: {step.get('description', 'No description')}")
        return state
    
    async def main(self, user_input: str):
        """
        Main orchestration method for multi-agent planning
        
        Args:
            user_input (str): User's original input
        """
        # Generate comprehensive prompt using the prompt generation agent
        prompt = await self.generate_prompt_for_agents(user_input)
        
        # Generate plans concurrently for planning agents
        agent_plans = await asyncio.gather(
            *[self.generate_agent_plan(f"Agent_{i}", prompt) 
              for i in range(3)]  # Dynamic agent count
        )
        
        # Calculate confidence score for each agent plan and store reasoning
        for plan in agent_plans:
            plan.confidence_score, reason = await self.calculate_confidence_score(plan)
            print(f"Agent {plan.agent_id} Confidence Score: {plan.confidence_score}")
            print(f"Reasoning: {reason}")
        
        # Evaluate and select the best plan using LLM
        best_plan = await self.evaluate_plans(agent_plans)
        
        # Print the selected plan
        print("\n--- Selected Plan ---")
        for step in best_plan.steps:
            print(f"Step {step['step']}: {step['action']}")
            print(f"Description: {step['description']}\n")

# Example usage
async def run_multi_agent_planner():
    planner = MultiAgentPlanner()
    
    # Get user input
    user_input = input("Enter your task: ")
    
    try:
        # Run the multi-agent planner
        await planner.main(user_input)
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(run_multi_agent_planner())
