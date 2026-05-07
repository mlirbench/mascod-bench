from langgraph.graph import StateGraph
from typing import TypedDict

from agents.root_agent.node import root_agent
from agents.mutation_agent.node import mutation_agent
from agents.execution_sandbox_agent.node import execution_sandbox_agent
from agents.llm_reasoning_agent.node import run as run_llm_reasoning
from agents.judge_agent.node import judge_agent

class GraphState(TypedDict, total=False):
    user_input: dict
    root_output: dict
    mutation_output: dict
    toExecutionAgent: dict
    llm_reasoning_output: dict
    judge_payload: dict
    judge_details: dict
    judge_input: dict
    Agents: dict
    logs: list[str]


BRIDGE_KEYS = {"toExecutionAgent", "toLLMReasoning"}


def _without_bridge_payloads(value):
    if isinstance(value, dict):
        return {
            key: _without_bridge_payloads(item)
            for key, item in value.items()
            if key not in BRIDGE_KEYS
        }
    if isinstance(value, list):
        return [_without_bridge_payloads(item) for item in value]
    return value


def root_node(state: GraphState):
    result = root_agent(state["user_input"])
    return {"root_output": result.dict()}

def mutation_node(state: GraphState):
    result = mutation_agent(state["root_output"])
    mutation_agent_output = result.get("Agents", {}).get("Mutation_Agent", {})
    return {
        "mutation_output": result,
        "toExecutionAgent": mutation_agent_output.get("toExecutionAgent"),
        "Agents": {
            **state.get("Agents", {}),
            **result.get("Agents", {}),
        },
    }

def execution_node(state: GraphState):
    return execution_sandbox_agent(state)

def llm_reasoning_node(state: GraphState):
    logs = state.get("logs", [])
    logs.append("LLM Reasoning Agent: Starting")

    payload = state["Agents"]["Execution_Agent"]["toLLMReasoning"]
    embedded_inputs = payload.get("embed_inputs", {})
    programA = payload.get("programA")
    programB = payload.get("programB")
    dialect = payload.get("dialect")
    mutation_context = (
        state.get("Agents", {})
        .get("Mutation_Agent", {})
        .get("context_results", {})
    )

    result = run_llm_reasoning(
        {
            "programA": programA,
            "programB": programB,
            "dialect": dialect,
            "embedded_inputs": embedded_inputs,
        }
    )
    reasoning_output = result.get("Agents", {}).get("LLM_Reasoning_Agent", {})

    logs.append(
        f"LLM Reasoning Agent: Finished {len(reasoning_output.get('results', []))} model result(s)"
    )

    return {
        **state,
        "llm_reasoning_output": reasoning_output,
        "Agents": {
            **state.get("Agents", {}),
            **result.get("Agents", {}),
        },
        "logs": logs,
    }

def judge_node(state: GraphState):
    logs = state.get("logs", [])
    logs.append("Judge Agent: Starting")

    cleaned_agents = _without_bridge_payloads(state.get("Agents", {}))
    mutation_agent_output = cleaned_agents.get("Mutation_Agent", {})
    execution_agent_output = cleaned_agents.get("Execution_Agent", {})
    reasoning_agent_output = cleaned_agents.get("LLM_Reasoning_Agent", {})

    mutation_context = mutation_agent_output.get("context_results", {})
    execution_results = execution_agent_output.get("execution_results", {})
    reasoning_results = reasoning_agent_output.get("results", [])

    base_judge_input = {
        "Agents": cleaned_agents,
        "programA": state.get("toExecutionAgent", {}).get("input_mlir"),
        "programB": state.get("toExecutionAgent", {}).get("mutated_mlir"),
        "dialect": state.get("toExecutionAgent", {}).get("dialect"),
        "metadata": mutation_context.get("metadata", {}),
        "mutation_rule": mutation_context.get("requirement", {}),
        "mutation_id": mutation_context.get("requirement", {}).get("id"),
        "mutation_kind": mutation_context.get("requirement", {}).get("description"),
        "source": "run-benchmark",
        **execution_results,
    }

    evaluations = []
    for prediction in reasoning_results:
        judge_state = {
            "judge_input": {
                "prediction": prediction,
                **base_judge_input,
            },
            "eval_model_name": prediction.get("model_key"),
            "logs": logs,
        }
        judge_result = judge_agent(judge_state)
        logs = judge_result.get("logs", logs)
        evaluations.append(
            {
                "model_key": prediction.get("model_key"),
                "llm_name": prediction.get("llm_name"),
                "provider": prediction.get("provider"),
                "judge_payload": judge_result.get("judge_payload"),
                "judge_details": judge_result.get("judge_details"),
            }
        )

    result = {
        "evaluations": evaluations,
        "evaluated_count": len(evaluations),
    }
    logs.append(f"Judge Agent: Finished {len(evaluations)} evaluation(s)")

    return {
        **state,
        "judge_payload": result,
        "judge_input": base_judge_input,
        "Agents": {
            **cleaned_agents,
            "Judge_Agent": result,
        },
        "logs": logs,
    }

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("root", root_node)
    builder.add_node("mutation", mutation_node)
    builder.add_node("execution", execution_node)
    builder.add_node("llm_reasoning", llm_reasoning_node)
    builder.add_node("judge", judge_node)

    builder.set_entry_point("root")
    builder.add_edge("root", "mutation")
    builder.add_edge("mutation", "execution")
    builder.add_edge("execution", "llm_reasoning")
    builder.add_edge("llm_reasoning","judge")

    builder.set_finish_point("judge")

    return builder.compile()
