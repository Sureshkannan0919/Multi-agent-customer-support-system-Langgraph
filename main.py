import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_from_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from policy_rag import PolicyRAG
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Base Models for Controlled Input/Output ---
class IntentClassification(BaseModel):
    """Controlled output for intent classification"""
    intent: str = Field(description="The classified intent", enum=["recommendation", "policy_support", "order_related"])
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)
    reasoning: str = Field(description="Brief reasoning for the classification")

class RecommendationResponse(BaseModel):
    """Controlled output for recommendation responses"""
    recommendation: str = Field(description="The main recommendation")
    category: str = Field(description="Category of recommendation (product, service, etc.)")
    confidence: float = Field(description="Confidence in the recommendation", ge=0, le=1)

class OrderResponse(BaseModel):
    """Controlled output for order-related responses"""
    action_required: str = Field(description="What action is needed")
    status: str = Field(description="Current status of the order/issue")
    next_steps: str = Field(description="Next steps for the customer")

class PolicyResponse(BaseModel):
    """Controlled output for policy responses"""
    answer: str = Field(description="Answer to the policy question")
    policy_reference: str = Field(description="Reference to relevant policy section")
    confidence: float = Field(description="Confidence in the answer", ge=0, le=1)

# --- Intent Classifier Tool ---
def classify_intent(query: str) -> IntentClassification:
    """
    Classifies the intent of the user query using controlled output.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are an intent classifier for a customer support system. "
                    "Classify the user query and provide a structured response with intent, confidence, and reasoning. "
                    "Intent must be one of: 'recommendation', 'policy_support', or 'order_related'."
        ),
        HumanMessage(content="{query}")
    ])
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    
    # Use structured output
    structured_chain = prompt | llm.with_structured_output(IntentClassification)
    result = structured_chain.invoke({"query": query})
    
    return result

# --- Policy Support Tool ---
def policy_support_tool(query: str) -> PolicyResponse:
    """
    Answers policy-related queries using the PolicyRAG system with controlled output.
    """
    policy_path = os.environ.get("POLICY_FILE_PATH", "policy.txt")
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    rag = PolicyRAG(policy_path, google_api_key)
    answer, metadata = rag.answer_query(query)
    
    # Create structured response
    return PolicyResponse(
        answer=answer,
        policy_reference=metadata.get("source", "general_policy") if metadata else "general_policy",
        confidence=0.9 if metadata else 0.7
    )

# --- Recommendation Tool ---
def recommendation_tool(query: str) -> RecommendationResponse:
    """
    Handles product/service recommendation queries with controlled output.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are a helpful customer support agent specializing in product and service recommendations. "
                    "Provide a structured recommendation response with the recommendation, category, and confidence level."
        ),
        HumanMessage(content="{query}")
    ])
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    structured_chain = prompt | llm.with_structured_output(RecommendationResponse)
    result = structured_chain.invoke({"query": query})
    
    return result

# --- Order Related Tool ---
def order_related_tool(query: str) -> OrderResponse:
    """
    Handles order-related queries with controlled output.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="You are a customer support agent specializing in order-related issues. "
                    "Provide a structured response with action required, status, and next steps."
        ),
        HumanMessage(content="{query}")
    ])
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    structured_chain = prompt | llm.with_structured_output(OrderResponse)
    result = structured_chain.invoke({"query": query})
    
    return result

# --- Tool Nodes ---
intent_classifier_node = ToolNode(
    tools=tools_from_function(classify_intent),
    name="IntentClassifier"
)
policy_support_node = ToolNode(
    tools=tools_from_function(policy_support_tool),
    name="PolicySupport"
)
recommendation_node = ToolNode(
    tools=tools_from_function(recommendation_tool),
    name="Recommendation"
)
order_related_node = ToolNode(
    tools=tools_from_function(order_related_tool),
    name="OrderRelated"
)

# --- StateGraph Construction ---
graph = StateGraph()

# Add nodes
graph.add_node("intent_classifier", intent_classifier_node)
graph.add_node("policy_support", policy_support_node)
graph.add_node("recommendation", recommendation_node)
graph.add_node("order_related", order_related_node)

# Edges: intent classifier routes to the right node
def route_by_intent(state):
    intent_result = state.get("intent_classifier")
    if intent_result and hasattr(intent_result, 'intent'):
        intent = intent_result.intent
    else:
        intent = "recommendation"  # fallback
        
    if intent == "policy_support":
        return "policy_support"
    elif intent == "order_related":
        return "order_related"
    else:
        return "recommendation"

graph.add_edge("intent_classifier", route_by_intent)
graph.add_edge("policy_support", END)
graph.add_edge("recommendation", END)
graph.add_edge("order_related", END)

# Set entry point
graph.set_entry_point("intent_classifier")

# Compile the agentic system
customer_support_agent = graph.compile()

# --- Example Usage ---
if __name__ == "__main__":
    print("Welcome to the Customer Support Agentic AI System!")
    print("Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.strip().lower() == "exit":
            break
        
        # The agent expects a dict as input state
        result = customer_support_agent.invoke({"query": user_query})
        
        # Display structured output based on the node that was executed
        if "intent_classifier" in result:
            intent_result = result["intent_classifier"]
            print(f"Intent: {intent_result.intent} (Confidence: {intent_result.confidence:.2f})")
            print(f"Reasoning: {intent_result.reasoning}")
        
        # Find the last non-classifier node's output and display it
        for node in ["policy_support", "order_related", "recommendation"]:
            if node in result and result[node]:
                response = result[node]
                print(f"\n{node.replace('_', ' ').title()} Response:")
                
                if hasattr(response, 'answer'):
                    print(f"Answer: {response.answer}")
                    print(f"Policy Reference: {response.policy_reference}")
                    print(f"Confidence: {response.confidence:.2f}")
                elif hasattr(response, 'recommendation'):
                    print(f"Recommendation: {response.recommendation}")
                    print(f"Category: {response.category}")
                    print(f"Confidence: {response.confidence:.2f}")
                elif hasattr(response, 'action_required'):
                    print(f"Action Required: {response.action_required}")
                    print(f"Status: {response.status}")
                    print(f"Next Steps: {response.next_steps}")
                else:
                    print(f"Response: {response}")
                break
