"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
from typing import List, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    fact_check_findings_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)
from open_deep_research.council import (
    council_vote_on_brief,
    CouncilConfig,
    log_council_decision,
)
from pydantic import BaseModel, Field as PydanticField

# Pydantic model for fact-checking findings
class FindingsReview(BaseModel):
    """Structured review of research findings for fact-checking."""
    
    decision: Literal["approve", "revise", "reject"] = PydanticField(
        description="approve=findings are factually grounded, revise=issues found that need fixing, reject=major fabrications detected"
    )
    confidence: float = PydanticField(
        ge=0, le=1,
        description="How confident are you in this assessment? 0-1"
    )
    issues_found: List[str] = PydanticField(
        description="List of specific issues found (fabricated names, impossible dates, uncited claims, etc.)"
    )
    suggested_fixes: List[str] = PydanticField(
        description="Specific recommendations to fix the issues"
    )
    reasoning: str = PydanticField(
        description="Overall assessment of the findings quality and factual accuracy"
    )

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["validate_brief"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to brief validation (council) with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    
    # Include council feedback if this is a revision attempt
    feedback_on_brief = state.get("feedback_on_brief", [])
    if feedback_on_brief:
        prompt_content += f"\n\nPREVIOUS FEEDBACK TO ADDRESS:\n{feedback_on_brief[-1]}"
    
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    # Use effective values (reduced in test mode)
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.get_effective_max_concurrent_research_units(),
        max_researcher_iterations=configurable.get_effective_max_researcher_iterations()
    )
    
    return Command(
        goto="validate_brief", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def validate_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor", "write_research_brief"]]:
    """Validate the research brief using the LLM Council and optionally human review.
    
    In this architecture, Council = Advisor, Human = Authority:
    - Council provides feedback and suggestions (not approve/reject decisions)
    - If review_mode != "none", human reviews brief + council feedback
    - Human can approve, edit, or ignore council feedback
    - Once human approves (or if review_mode == "none"), proceed to research
    
    Args:
        state: Current agent state with research brief
        config: Runtime configuration with council and review settings
        
    Returns:
        Command to proceed to research (after human approval if needed)
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Get current brief (use human-approved version if available)
    brief = state.get("human_approved_brief") or state.get("research_brief", "")
    if not brief:
        # No brief to validate - proceed anyway
        return Command(goto="research_supervisor")
    
    # Check if human already approved (resuming after interrupt)
    if state.get("human_approved_brief"):
        # Human has approved - proceed to research without re-validation
        print(f"[REVIEW] Using human-approved brief. Proceeding to research.")
        return Command(goto="research_supervisor")
    
    # Get council feedback (advisory only, not approve/reject)
    council_feedback = ""
    if configurable.use_council:
        council_config = CouncilConfig(
            models=configurable.council_models,
            min_consensus_for_approve=configurable.council_min_consensus,
            max_revision_rounds=configurable.council_max_revisions,
        )
        
        verdict = await council_vote_on_brief(brief, council_config)
        log_council_decision(verdict)
        
        # Format council feedback for human review
        council_feedback = f"""
COUNCIL FEEDBACK (Advisory):
Decision: {verdict.decision.upper()}
Consensus: {verdict.consensus_score:.0%}

{verdict.synthesized_feedback}
"""
        print(f"\n{'='*60}")
        print(f"COUNCIL FEEDBACK (Advisory - Human has final say)")
        print(f"Decision: {verdict.decision.upper()}")
        print(f"Consensus: {verdict.consensus_score:.0%}")
        print(f"{'='*60}\n")
    
    # Human review checkpoint
    if configurable.review_mode != "none":
        # Format the review request for human
        review_request = f"""
═══════════════════════════════════════════════════════════════
HUMAN REVIEW REQUIRED: Research Brief
═══════════════════════════════════════════════════════════════

RESEARCH BRIEF:
{brief}

{council_feedback if council_feedback else "(Council review disabled)"}

═══════════════════════════════════════════════════════════════
OPTIONS:
1. Reply with "approve" to proceed with this brief
2. Reply with your edited version of the brief to use instead
3. Reply with "ignore" to proceed without council suggestions
═══════════════════════════════════════════════════════════════
"""
        # Interrupt for human review - execution pauses here
        human_response = interrupt(review_request)
        
        # DEBUG: Log what we received from interrupt
        print(f"\n{'='*60}")
        print(f"[DEBUG BRIEF] Raw interrupt response type: {type(human_response)}")
        print(f"[DEBUG BRIEF] Raw interrupt response repr: {repr(human_response)[:200]}")
        print(f"{'='*60}\n")
        
        try:
            # Process human response with error handling
            response_str = str(human_response).strip().strip('"\'')  # Strip quotes for JSON/YAML
            response_lower = response_str.lower()
            
            print(f"[DEBUG BRIEF] After processing: '{response_str}' (len={len(response_str)})")
            
            # Known commands
            approve_commands = ["approve", "ok", "yes", "y", "proceed", "continue", "go", "accept"]
            ignore_commands = ["ignore", "skip", "no council", "dismiss"]
            
            if response_lower in approve_commands or response_lower in ignore_commands:
                # Human approved the brief as-is
                print(f"[REVIEW] Human approved brief with command: {response_lower}")
                return Command(
                    goto="research_supervisor",
                    update={
                        "human_approved_brief": brief,
                        "council_brief_feedback": council_feedback
                    }
                )
            elif len(response_str) < 50:
                # Short response that's not a known command - likely a mistake
                # Default to approve with warning
                print(f"[REVIEW] Unknown short response '{response_str}'. Defaulting to approve. "
                      f"Use 'approve', 'ignore', or provide a full edited brief (50+ chars).")
                return Command(
                    goto="research_supervisor",
                    update={
                        "human_approved_brief": brief,
                        "council_brief_feedback": council_feedback
                    }
                )
            else:
                # Human provided an edited brief (substantial text) - use their version
                print(f"[REVIEW] Human provided edited brief ({len(response_str)} chars).")
                return Command(
                    goto="research_supervisor",
                    update={
                        "human_approved_brief": response_str,
                        "council_brief_feedback": council_feedback,
                        "research_brief": response_str,
                        # Update supervisor messages with new brief
                        "supervisor_messages": {
                            "type": "override",
                            "value": [
                                SystemMessage(content=lead_researcher_prompt.format(
                                    date=get_today_str(),
                                    max_concurrent_research_units=configurable.get_effective_max_concurrent_research_units(),
                                    max_researcher_iterations=configurable.get_effective_max_researcher_iterations()
                                )),
                                HumanMessage(content=response_str)
                            ]
                        }
                    }
                )
        except Exception as e:
            # If anything goes wrong, log it and default to approve
            print(f"[ERROR BRIEF] Exception processing interrupt: {e}")
            print(f"[ERROR BRIEF] Response was: {repr(human_response)[:200]}")
            import traceback
            traceback.print_exc()
            # Fallback: default to approve
            return Command(
                goto="research_supervisor",
                update={
                    "human_approved_brief": brief,
                    "council_brief_feedback": council_feedback
                }
            )
    
    # No human review required (review_mode == "none")
    # In auto mode, still use council decision for routing
    if configurable.use_council:
        revision_count = state.get("council_revision_count", 0)
        
        if verdict.decision == "approve":
            return Command(
                goto="research_supervisor",
                update={"council_brief_feedback": council_feedback}
            )
        elif verdict.decision == "reject" or revision_count >= configurable.council_max_revisions:
            print(f"[COUNCIL] Auto-mode: Proceeding after {revision_count} revisions.")
            return Command(
                goto="research_supervisor",
                update={"council_brief_feedback": council_feedback}
            )
        else:
            return Command(
                goto="write_research_brief",
                update={
                    "feedback_on_brief": [verdict.synthesized_feedback],
                    "council_revision_count": revision_count + 1,
                    "council_brief_feedback": council_feedback
                }
            )
    
    # No council, no human review - just proceed
    return Command(goto="research_supervisor")


async def validate_findings(state: AgentState, config: RunnableConfig) -> Command[Literal["final_report_generation", "research_supervisor"]]:
    """Fact-check research findings and flag issues for human review.
    
    In this architecture, Council = Advisor, Human = Authority:
    - Council fact-checks findings and FLAGS issues (doesn't auto-reject)
    - Issues are stored in state for human review
    - If review_mode == "full" and issues found, human reviews before report
    - Human can: approve anyway, request re-research, or proceed
    
    Args:
        state: Current agent state with research notes
        config: Runtime configuration with fact-check and review settings
        
    Returns:
        Command to proceed to final report or loop back (if human requests re-research)
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Skip fact-checking if disabled
    if not configurable.use_findings_council:
        return Command(goto="final_report_generation")
    
    # Get research findings from notes
    notes = state.get("notes", [])
    if not notes:
        # No findings to validate - proceed anyway
        return Command(goto="final_report_generation")
    
    findings_text = "\n\n".join(notes)
    
    # Configure the fact-checking model
    model_config = {
        "model": configurable.council_models[0] if configurable.council_models else "openai:gpt-4.1",
        "max_tokens": 4096,
        "api_key": get_api_key_for_model(configurable.council_models[0] if configurable.council_models else "openai:gpt-4.1", config),
        "tags": ["langsmith:fact_check"]
    }
    
    fact_check_model = (
        configurable_model
        .with_structured_output(FindingsReview)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Generate fact-check prompt
    prompt = fact_check_findings_prompt.format(
        date=get_today_str(),
        findings=findings_text
    )
    
    try:
        review: FindingsReview = await fact_check_model.ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        # If fact-check fails, log and proceed
        print(f"[FACT-CHECK] Error during fact-check: {e}. Proceeding to report.")
        return Command(goto="final_report_generation")
    
    # Format flagged issues
    flagged_issues = []
    if review.issues_found:
        flagged_issues = review.issues_found
    
    # Log the fact-check results
    print(f"\n{'='*60}")
    print(f"FACT-CHECK RESULTS (Advisory - Human has final say)")
    print(f"Assessment: {review.decision.upper()}")
    print(f"Confidence: {review.confidence:.0%}")
    if flagged_issues:
        print(f"Issues Flagged: {len(flagged_issues)}")
        for issue in flagged_issues[:3]:
            print(f"  ⚠ {issue[:100]}...")
    else:
        print(f"No issues flagged.")
    print(f"{'='*60}\n")
    
    # Log detailed feedback (non-blocking, for offline review)
    if flagged_issues:
        print(f"\n{'='*60}")
        print(f"[FACT-CHECK] ⚠️ {len(flagged_issues)} issues flagged (advisory):")
        for issue in flagged_issues[:5]:
            print(f"  - {issue[:100]}...")
        if review.suggested_fixes:
            print(f"\n[FACT-CHECK] Suggested fixes:")
            for fix in review.suggested_fixes[:3]:
                print(f"  → {fix[:100]}...")
        print(f"\n[FACT-CHECK] Reasoning: {review.reasoning[:200]}...")
        print(f"[FACT-CHECK] Flags stored in state for offline review.")
        print(f"{'='*60}\n")
    
    # Always proceed to final report (non-blocking)
    # Flagged issues stored in state for later review
    return Command(
        goto="final_report_generation",
        update={"flagged_issues": flagged_issues if flagged_issues else []}
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase (use effective values for test mode)
    exceeded_allowed_iterations = research_iterations > configurable.get_effective_max_researcher_iterations()
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion (use effective values for test mode)
            max_units = configurable.get_effective_max_concurrent_research_units()
            allowed_conduct_research_calls = conduct_research_calls[:max_units]
            overflow_conduct_research_calls = conduct_research_calls[max_units:]
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {max_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools, use effective values for test mode)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.get_effective_max_react_tool_calls()
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with optional human review.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report. If review_mode == "full", the report
    is presented to the human for approval before completion.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    generated_report = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            generated_report = final_report.content
            break  # Success - exit retry loop
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Check if report generation succeeded
    if generated_report is None:
        return {
            "final_report": "Error generating final report: Maximum retries exceeded",
            "messages": [AIMessage(content="Report generation failed after maximum retries")],
            **cleared_state
        }
    
    # Step 4: Human review checkpoint (if review_mode == "full")
    if configurable.review_mode == "full":
        # Get any flagged issues from fact-check
        flagged_issues = state.get("flagged_issues", [])
        issues_section = ""
        if flagged_issues:
            issues_section = f"""
⚠️ FLAGGED ISSUES FROM FACT-CHECK:
{chr(10).join(f'  ⚠ {issue}' for issue in flagged_issues)}
"""
        
        # Format review request
        review_request = f"""
═══════════════════════════════════════════════════════════════
HUMAN REVIEW REQUIRED: Final Report
═══════════════════════════════════════════════════════════════
{issues_section}
GENERATED REPORT:
{generated_report[:5000]}{'... [truncated for review]' if len(generated_report) > 5000 else ''}

═══════════════════════════════════════════════════════════════
OPTIONS:
1. Reply with "approve" to accept this report
2. Reply with specific feedback to regenerate with changes
3. Reply with your own edited version of the report
═══════════════════════════════════════════════════════════════
"""
        # Interrupt for human review
        human_response = interrupt(review_request)
        
        # DEBUG: Log what we received from interrupt
        print(f"\n{'='*60}")
        print(f"[DEBUG REPORT] Raw interrupt response type: {type(human_response)}")
        print(f"[DEBUG REPORT] Raw interrupt response repr: {repr(human_response)[:200]}")
        print(f"{'='*60}\n")
        
        try:
            # Process human response with error handling
            response_str = str(human_response).strip().strip('"\'')  # Strip quotes for JSON/YAML
            response_lower = response_str.lower()
            
            print(f"[DEBUG REPORT] After processing: '{response_str[:100]}...' (len={len(response_str)})")
            
            # Known commands
            approve_commands = ["approve", "ok", "yes", "y", "proceed", "continue", "go", "accept", "done", "good"]
            
            if response_lower in approve_commands:
                # Human approved - return the report
                print(f"[REVIEW] Human approved final report with command: {response_lower}")
                return {
                    "final_report": generated_report, 
                    "messages": [AIMessage(content=generated_report)],
                    **cleared_state
                }
            elif len(response_str) > 200:
                # Human provided their own edited version (substantial text)
                print(f"[REVIEW] Human provided edited report ({len(response_str)} chars).")
                return {
                    "final_report": response_str, 
                    "messages": [AIMessage(content=response_str)],
                    **cleared_state
                }
            elif len(response_str) < 20:
                # Very short unknown response - default to approve with warning
                print(f"[REVIEW] Unknown short response '{response_str}'. Defaulting to approve. "
                      f"Use 'approve' to accept, or provide substantial edits (200+ chars).")
                return {
                    "final_report": generated_report, 
                    "messages": [AIMessage(content=generated_report)],
                    **cleared_state
                }
            else:
                # Medium-length response - likely feedback, proceed with original
                # (regeneration would require another LLM call which is complex)
                print(f"[REVIEW] Human feedback noted: '{response_str[:50]}...'. Proceeding with original report. "
                      f"To edit, provide the full report text (200+ chars).")
                return {
                    "final_report": generated_report, 
                    "messages": [AIMessage(content=generated_report)],
                    **cleared_state
                }
        except Exception as e:
            # If anything goes wrong, log it and default to approve
            print(f"[ERROR REPORT] Exception processing interrupt: {e}")
            print(f"[ERROR REPORT] Response was: {repr(human_response)[:200]}")
            import traceback
            traceback.print_exc()
            # Fallback: default to returning the report
            return {
                "final_report": generated_report, 
                "messages": [AIMessage(content=generated_report)],
                **cleared_state
            }
    
    # No human review required - return the report
    return {
        "final_report": generated_report, 
        "messages": [AIMessage(content=generated_report)],
        **cleared_state
    }


async def verify_claims(state: AgentState, config: RunnableConfig):
    """Health check: verify claims in final report against preserved sources.
    
    This runs AFTER the report is generated. The report is already in state,
    this adds verification as a quality check layer.
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Skip if disabled
    if not configurable.use_claim_verification:
        print("[VERIFY] Claim verification disabled, skipping.")
        return {}
    
    # Get sources from store
    from open_deep_research.utils import get_stored_sources
    sources = await get_stored_sources(config)
    
    if not sources:
        print("[VERIFY] No sources found in store, skipping verification.")
        return {"verification_result": None}
    
    print(f"[VERIFY] Starting verification with {len(sources)} sources...")
    
    # Run verification
    from open_deep_research.verification import verify_report
    result = await verify_report(
        final_report=state.get("final_report", ""),
        sources=sources,
        config=config
    )
    
    # Log summary
    summary = result["summary"]
    print(f"[VERIFY] Complete: {summary['supported']}/{summary['total_claims']} supported")
    print(f"[VERIFY] Confidence: {summary['overall_confidence']:.0%}")
    
    # Log warnings (flagged for offline review, no blocking interrupt)
    if summary.get("warnings"):
        print(f"[VERIFY] ⚠️ {len(summary['warnings'])} claims flagged for review:")
        for w in summary["warnings"][:5]:
            print(f"  - {w}")
        print("[VERIFY] Flags included in verification_result for offline review")
    
    # Log data issues if present
    if summary.get("data_issues"):
        print(f"[VERIFY] 📋 {len(summary['data_issues'])} data issues detected:")
        for issue in summary["data_issues"][:3]:
            print(f"  - {issue}")
    
    return {"verification_result": result}


# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("validate_brief", validate_brief)                 # Council 1: Brief validation
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("validate_findings", validate_findings)           # Council 2: Fact-check findings
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase
deep_researcher_builder.add_node("verify_claims", verify_claims)                   # Claim verification health check

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "validate_findings")       # Research to fact-check
deep_researcher_builder.add_edge("final_report_generation", "verify_claims")       # Report to verification
deep_researcher_builder.add_edge("verify_claims", END)                             # Final exit point
# Note: write_research_brief -> validate_brief, validate_brief routing, and validate_findings routing handled by Command returns

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()