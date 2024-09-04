# LangChain vs LangGraph: Computational Models

## LangChain: Directed Acyclic Computation

LangChain primarily uses a Directed Acyclic Graph (DAG) model for its workflows. This means that the flow of operations moves in one direction without cycles.

### Example: Simple Question-Answering Chain

```python
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# Define a template
template = """Question: {question}

Answer: Let's approach this step-by-step:"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Create the LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=OpenAI(temperature=0),
)

# Run the chain
question = "What is the capital of France?"
response = llm_chain.run(question)
print(response)
```

In this LangChain example:
1. The question is input.
2. It's processed through the prompt template.
3. The LLM generates a response.
4. The answer is output.

This is a linear, one-directional flow without any loops or cycles.

## LangGraph: Cyclic Computation

LangGraph allows for cyclic computations, meaning the workflow can loop back to previous steps or continue processing based on certain conditions.

### Example: Iterative Task Solving Agent

```python
from langchain.chat_models import ChatOpenAI
from langgraph.graph import Graph, END
from langchain.prompts import ChatPromptTemplate

# Define the agent's state
class AgentState(TypedDict):
    message: str
    task_list: List[str]
    completed_tasks: List[str]

# Function to process tasks
def process_task(state: AgentState) -> AgentState:
    current_task = state["task_list"][0]
    prompt = ChatPromptTemplate.from_template(
        "Complete the task: {task}. Current progress: {progress}"
    )
    llm = ChatOpenAI()
    response = llm.invoke(prompt.format(task=current_task, progress=state["message"]))
    
    state["message"] += f"\nCompleted: {current_task}"
    state["completed_tasks"].append(current_task)
    state["task_list"] = state["task_list"][1:]
    
    return state

# Function to check if tasks are complete
def tasks_complete(state: AgentState) -> bool:
    return len(state["task_list"]) == 0

# Create the graph
workflow = Graph()

# Add the processing node
workflow.add_node("process_task", process_task)

# Add the conditional edge
workflow.add_conditional_edges(
    "process_task",
    tasks_complete,
    {
        True: END,
        False: "process_task"  # Loop back if tasks remain
    }
)

# Set the entry point
workflow.set_entry_point("process_task")

# Compile the graph
app = workflow.compile()

# Run the graph
initial_state = {
    "message": "Starting tasks",
    "task_list": ["Task 1", "Task 2", "Task 3"],
    "completed_tasks": []
}

final_state = app.invoke(initial_state)
print(final_state)
```

In this LangGraph example:
1. We define a state that includes a task list and completed tasks.
2. The `process_task` function handles a single task.
3. The `tasks_complete` function checks if all tasks are done.
4. The graph is set up to repeatedly process tasks until they're all complete.
5. If tasks remain, it loops back to `process_task`, creating a cycle.

This cyclic structure allows for iterative processing, where the agent can continue working on tasks until a condition is met, unlike the linear flow in the LangChain example.

