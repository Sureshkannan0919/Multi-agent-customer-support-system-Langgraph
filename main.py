import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="langchain_google_genai._function_utils")

from datetime import datetime
from typing import Dict, Any, Literal, Optional, Callable, TypedDict, Annotated
from pydantic import BaseModel, Field 

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import AnyMessage, add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, Runnable
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode


from src.custom_tool import fetch_user_details, fetch_order_details ,search_product_online
from src.policy_rag import lookup_policy
from src.product_recomentaion_rag import search_product
import shutil
import uuid



os.environ["GOOGLE_API_KEY"] = "GEMINI-API-KEY"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "Product_recommender",
                "product_comparator",
                "order_related",
                "policy_support",  
            ]
        ],
        update_dialog_stack,
    ]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }

class ToProductRecommender(BaseModel):
    request:str = Field(description="The request to be sent to the product recommender tool.")

class ToProductComparator(BaseModel):
    request:str = Field(description="The request to be sent to the product comparator tool.")

class ToOrderRelated(BaseModel):
    request:str = Field(description="The request to be sent to the order related tool.")

class ToPolicySupport(BaseModel):
    request:str = Field(description="The request to be sent to the policy support tool.")



assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Start greeting to the user. User info provided below:"
            "You are a helpful e-commerce assistant for a sports products store (e.g., shoes, skates, apparel, accessories). "
            " Use the provided tools to search product catalog, sizing guides, inventory, shipping/returns policies, and order information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, broaden the query, try related categories, and suggest close matches before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

primary_assistant_tools = [fetch_user_details]

assistant_runnable = assistant_prompt | llm.bind_tools(
    primary_assistant_tools + [
        ToProductRecommender,
        ToProductComparator,
        ToOrderRelated,
        ToPolicySupport,
        CompleteOrEscalate
    ]
)


product_recommender = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are an AI Product Recommender for our sports store.\n\n"
        "� ***CRITICAL RULES**:\n"
        "- ALWAYS use the search_product tool FIRST before making any recommendations\n"
        "- NEVER recommend products that are not found in our store inventory\n"
        "- NEVER generate fake or imaginary products\n"
        "- ONLY recommend products returned by the search_product tool\n"
        "- If no products are found, inform the user and suggest they try different search terms\n\n"
        
        "🛍️ **WORKFLOW**:\n"
        "1. ALWAYS use search_product tool with user's query first\n"
        "2. Wait for the tool response with actual product data\n"
        "3. Parse the returned product information (name, category, price, description)\n"
        "4. Present recommendations using ONLY the exact data from tool response\n"
        "5. If tool returns 'No products found', inform user and suggest different search terms\n"
        "6. NEVER make up product details not provided by the tool\n\n"
        
        "🔧 **TOOLS AVAILABLE**:\n"
        "- search_product: Search our store inventory (MUST USE THIS FIRST)\n"
        "- search_product_online: For additional online comparison (optional)\n\n"
        
        "📋 **RESPONSE FORMAT** (Only after using search_product tool):\n\n"
        "🛍️ **Product Recommendations from Our Store**\n\n"
        "Based on your search, here are the best matches from our inventory:\n\n"
        "**Top Recommendation**: [Product name from search results]\n\n"
        "**Category**: [Category from search results]\n\n"
        "**Price**: [Price from search results]\n\n"
        "**Description**: [Description from search results]\n\n"
        "**✅ Why I Recommend This**: [Explain why this product matches user needs]\n\n"
        "**⚠️ Things to Consider**: [Any limitations or considerations]\n\n"
        "**Alternative Option** (if multiple products found):\n\n"
        "**Product**: [Second product name]\n\n"
        "**Category**: [Category]\n\n"
        "**Price**: [Price]\n\n"
        "**Availability**: ✅ In stock at our store\n\n"
        "Would you like me to search for more options or compare with online alternatives?\n\n"
        
        "Current user info:\n<User>\n{user_info}\n</User>\n"
        "Current time: {time}",
        ),
        ("placeholder","{messages}"),
    ]
).partial(time=datetime.now)


product_search_tools = [search_product]

product_recommender_runnable = product_recommender | llm.bind_tools([search_product, CompleteOrEscalate])




product_comparator = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=
         "⚖️ **PRODUCT COMPARATOR MODE** - When users want to compare specific products:\n"
        "- Identify products for comparison based on user request\n"
        "- search online for similar product"
        "- if same product not found in online try to search similar product (eg:'i want to buy professional inline skate for my child','search online product': query = professional inline skate)"
        "- Gather detailed specifications from all sources\n"
        "- Create comprehensive comparison matrices\n"
        "- Analyze price differences across platforms\n"
        "- Compare user reviews and ratings\n"
        "- Provide clear winner recommendations\n\n"

        "**For COMPARATOR MODE:**\n"
        "⚖️ **Product Comparison Analysis**\n\n"
        "| Specification | Product A | Product B | Product C |\n"
        "|---------------|-----------|-----------|----------|\n"
        "| Price (Store) | ₹X | ₹Y | ₹Z |\n"
        "| Price (Amazon) | ₹X | ₹Y | ₹Z |\n"
        "| Price (Flipkart) | ₹X | ₹Y | ₹Z |\n"
        "| User Rating | X/5 ⭐ | Y/5 ⭐ | Z/5 ⭐ |\n"
        "| [Key Spec 1] | Value | Value | Value |\n"
        "| [Key Spec 2] | Value | Value | Value |\n\n"
        
        "**🏆 Best Overall**: [Product] - [Reasoning]\n\n"
        "**💰 Best Value**: [Product] - [Price-performance analysis]\n\n"
        "**🔥 Best Features**: [Product] - [Feature advantages]\n\n"
        "**📊 Summary**: [Brief recommendation]\n\n"
    
        "Current user info:\n<User>\n{user_info}\n</User>\n"
        "Current time: {time}",
        ),
        ("placeholder","{messages}"),
    ]

).partial(time=datetime.now)

search_similar_product = [search_product_online]

product_comparator_runnable = product_comparator | llm.bind_tools([search_product_online, CompleteOrEscalate])


policy_support = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful e-commerce assistant for a sports products store (e.g., shoes, skates, apparel, accessories). "
            "summarize the retrieved output and suggest the best deal"
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, broaden the query, try related categories, and suggest close matches before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "use product recommendation tool to get products and create context according to product recommendation tool results"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

policy_search_tool = [lookup_policy]

policy_support_runnable = policy_support | llm.bind_tools([lookup_policy, CompleteOrEscalate])

order_related = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a helpful e-commerce assistant for a sports products store (e.g., shoes, skates, apparel, accessories). "
            
            "CRITICAL RULES:\n"
            "- NEVER ask the user for their email address under any circumstances\n"
            "- The user's email is already provided in the user_info below - use it directly with your tools\n"
            "- Always use the available tools to search for order history, product information, and policies\n"
            "- Use search order tool and summarize the order details show only order details\n"
    

            "SEARCH STRATEGY:\n"
            "- tools and details are already provided in the user_info below - use it directly with your tools\n"
            "- use tools before reply for the user\n"
            "- Use the provided tools for: order status, billing address, total amount, shipping/returns policies, and order information\n"
            
            "USER CONTEXT:\n"
            "<User>\n{user_info}\n</User>\n"
            "The user_info contains the username and email - use the email directly with order search tools.\n"
            
            "Current time: {time}\n"
            
            "Always use tools to search for products and create comprehensive responses based on the search results.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now)

order_search_tool = [fetch_user_details,fetch_order_details]
order_related_runnable = order_related | llm.bind_tools([fetch_user_details,fetch_order_details, CompleteOrEscalate])

def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
        }

    return entry_node

def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    
    # Clear the dialog state to prevent infinite loops
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


builder = StateGraph(State)

def user_info(state: State):
    return {"user_info": fetch_user_details.invoke({})}

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

builder.add_node("fetch_user_info", user_info)
builder.add_edge(START, "fetch_user_info")
builder.add_node("product_search_tools", create_tool_node_with_fallback(product_search_tools))
builder.add_node("enter_product_recommender", create_entry_node("product recommender", "Product_recommender"))
builder.add_node("product_recommender", Assistant(product_recommender_runnable))
builder.add_edge("enter_product_recommender", "product_recommender")
builder.add_edge("product_search_tools", "product_recommender")
builder.add_node("leave_skill", pop_dialog_state)

def route_to_product_recommender(state: State):
    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return END
    
    # Check if user wants to cancel/escalate
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    
    return "product_search_tools"

builder.add_conditional_edges("product_recommender", route_to_product_recommender, ["product_search_tools", "leave_skill", END])

builder.add_edge("leave_skill", "primary_assistant")

builder.add_node("enter_product_comparator",create_entry_node("product_comparator","product_comparator"))
builder.add_node("product_comparator",Assistant(product_comparator_runnable))
builder.add_node("search_similar_product", create_tool_node_with_fallback(search_similar_product))
builder.add_edge("enter_product_comparator", "product_comparator")
builder.add_edge("search_similar_product", "product_comparator")
def route_to_product_comparator(state: State):
    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return END

    # Check if user wants to cancel/escalate
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"

    return "search_similar_product"
builder.add_conditional_edges("product_comparator", route_to_product_comparator, ["search_similar_product", "leave_skill", END])


builder.add_node("enter_order_related", create_entry_node("order related", "order_related"))
builder.add_node("order_related", Assistant(order_related_runnable))
builder.add_node("search_order_tools", create_tool_node_with_fallback(order_search_tool))
builder.add_edge("enter_order_related", "order_related")
builder.add_edge("search_order_tools", "order_related")

def route_to_order_support(state: State):
    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return END
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    # Route to the tool node when an order tool is called
    return "search_order_tools"

builder.add_conditional_edges("order_related", route_to_order_support, ["search_order_tools", "leave_skill", END])

builder.add_node("enter_policy_support", create_entry_node("policy support", "policy_support"))
builder.add_node("policy_support", Assistant(policy_support_runnable))
builder.add_node("search_policy_tools", create_tool_node_with_fallback(policy_search_tool))
builder.add_edge("enter_policy_support", "policy_support")
builder.add_edge("search_policy_tools", "policy_support")

def route_to_policy_support(state: State):
    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return END
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    # Route to the tool node when a policy tool is called
    return "search_policy_tools"

builder.add_conditional_edges("policy_support", route_to_policy_support, ["search_policy_tools", "leave_skill", END])

builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node("primary_assistant_tools", ToolNode(primary_assistant_tools))

def route_primary_assistant(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        tool_name = tool_calls[0]["name"]
        if tool_name == ToPolicySupport.__name__:
            return "enter_policy_support"
        elif tool_name == ToOrderRelated.__name__:
            return "enter_order_related"
        elif tool_name == ToProductRecommender.__name__:
            return "enter_product_recommender"
        elif tool_name == ToProductComparator.__name__:
            return "enter_product_comparator"
        return "primary_assistant_tools"
    return END

builder.add_conditional_edges("primary_assistant", route_primary_assistant, [
    "enter_product_recommender",
    "enter_product_comparator",
    "enter_order_related", 
    "enter_policy_support",
    "primary_assistant_tools",
    END,
])

builder.add_edge("primary_assistant_tools", "primary_assistant")

def route_to_workflow(state: State) -> Literal["primary_assistant"]:
    """Route to the primary assistant after fetching user info."""
    return "primary_assistant"

builder.add_conditional_edges("fetch_user_info", route_to_workflow, ["primary_assistant"])

memory = InMemorySaver()
graph = builder.compile(
    checkpointer=memory,
    )

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's information
        "uid": "ARaZlmY5mJRaEl3JaTKxes7oSVj2",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)



_printed = set()

def stream_graph(user_input: str):
    final_output = []
    events = graph.stream(
        {"messages": ("user", user_input)}, config, stream_mode="values"
    )
    for event in events:
        _print_event(event,_printed)
        


def chat_response(user_input: str):
    content_by_id = {}
    last_stop_message = None
    last_stop_content = ""
    _message = []
    events = graph.stream(
        {"messages": ("user", user_input)}, config, stream_mode="messages"
    )
    
    for event in events:
        message, metadata = event
        
        if hasattr(message, 'id') and hasattr(message, 'content'):
            msg_id = message.id
            
            # Combine content by ID
            if msg_id not in content_by_id:
                content_by_id[msg_id] = ""
            content_by_id[msg_id] += message.content
            
            # Check if this message has finish_reason = 'STOP'
            finish_reason = None
            if hasattr(message, 'response_metadata'):
                finish_reason = message.response_metadata.get('finish_reason')
            elif 'response_metadata' in metadata:
                finish_reason = metadata['response_metadata'].get('finish_reason')
            
            # If this chunk has finish_reason = 'STOP', it's our final message
            if finish_reason == 'STOP':
                last_stop_message = message
                last_stop_content = content_by_id[msg_id]
                if last_stop_content:
                    _message.append(last_stop_content)
                    print("content:  ",last_stop_content) # Get the complete combined content
    
    # Return the complete content of the last message with finish_reason = 'STOP'
    #return last_stop_content, last_stop_message
    return _message[-1]


# def export_graph():
#     """Export the graph for visualization purposes"""
#     return graph

# def main():
#     """Main function to run the interactive chat"""
#     while True:
#         user_input = input("User: ")
#         stream_graph(user_input)
#         # try:
#         #     user_input = input("User: ")
#         #     if user_input.lower(
#         #     ) in ["quit", "exit", "q"]:
#         #         print("Goodbye!")
#         #         break
#         #     stream_graph(user_input)
           


#         # except:
#         #     # fallback if input() is not available (e.g., in notebooks)
#         #     user_input = input("User: ")
#         #     stream_graph(user_input)
#         #     break

# if __name__ == "__main__":
#     main()


