# Tools Architecture: How Actions Flow from LLM to Browser

## Diagram 1: The 4-Layer Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LAYER 4: LLM PROVIDERS                          │
│                                                                         │
│   anthropic/chat.py    openai/chat.py    google/chat.py    groq/...    │
│                                                                         │
│   Each provider converts the AgentOutput Pydantic schema into its      │
│   native format (Anthropic tool_use, OpenAI structured output, etc.)   │
│   and parses the LLM response back into an AgentOutput instance.       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    AgentOutput (Pydantic model)
                    ┌─────────────────────────┐
                    │ thinking: str            │
                    │ evaluation: str          │
                    │ memory: str              │
                    │ next_goal: str           │
                    │ action: list[ActionModel]│◄── list of 1-5 actions
                    └────────────┬─────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────┐
│                        LAYER 3: AGENT                                   │
│                        agent/service.py                                  │
│                                                                         │
│   Agent._get_next_action()  ──► llm.ainvoke(messages, AgentOutput)      │
│                                         │                               │
│                                         ▼                               │
│   Agent.multi_act(actions)  ◄── parsed.action = [ActionModel, ...]      │
│         │                                                               │
│         ▼                                                               │
│   tools.act(action, browser_session=..., sensitive_data=..., ...)       │
└─────────┬───────────────────────────────────────────────────────────────┘
          │
          │  action = {"click": {"index": 5}}
          │  + injected deps (browser_session, llm, file_system, ...)
          │
┌─────────┼───────────────────────────────────────────────────────────────┐
│         ▼    LAYER 2: REGISTRY + ACTION FUNCTIONS                       │
│                                                                         │
│   tools/service.py          tools/registry/service.py                   │
│                                                                         │
│   Tools.act()               Registry.execute_action()                   │
│     │                          │                                        │
│     ├─ extract action name     ├─ validate params with Pydantic model   │
│     ├─ extract params dict     ├─ replace <secret> placeholders         │
│     └─ delegate to registry    ├─ inject special params                 │
│                                └─ call the actual async function        │
│                                                                         │
│   Registered functions:                                                 │
│     navigate(params, browser_session) ──► CDP NavigateToUrlEvent        │
│     click(params, browser_session)    ──► CDP ClickElementEvent         │
│     input(params, browser_session)    ──► CDP TypeTextEvent             │
│     scroll(params, browser_session)   ──► CDP ScrollEvent               │
│     extract(params, browser_session, page_extraction_llm) ──► LLM call  │
│     done(params, browser_session, file_system)                          │
│     ...20+ more actions                                                 │
└─────────┬───────────────────────────────────────────────────────────────┘
          │
          │  ActionResult(extracted_content=..., error=..., is_done=...)
          │
┌─────────┼───────────────────────────────────────────────────────────────┐
│         ▼    LAYER 1: PYDANTIC PARAM MODELS                             │
│                                                                         │
│   tools/views.py                                                        │
│                                                                         │
│   ClickElementAction       NavigateAction      InputTextAction          │
│   ┌──────────────────┐    ┌──────────────┐    ┌─────────────────┐      │
│   │ index: int|None  │    │ url: str     │    │ index: int      │      │
│   │ coordinate_x:    │    │ new_tab: bool│    │ text: str       │      │
│   │   int|None       │    └──────────────┘    │ clear: bool=True│      │
│   │ coordinate_y:    │                        └─────────────────┘      │
│   │   int|None       │    ScrollAction         DoneAction              │
│   └──────────────────┘    ┌──────────────┐    ┌─────────────────┐      │
│                           │ down: bool   │    │ text: str       │      │
│   These models are the    │ pages: float │    │ success: bool   │      │
│   SCHEMA SOURCE OF TRUTH  │ index: int?  │    └─────────────────┘      │
│   for both LLM and user.  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
```


## Diagram 2: How the LLM Sees Actions (Function Calling Schema)

```
Registry.create_action_model()  ←── called once at Agent startup
         │
         │  Iterates all RegisteredActions, creates one model per action:
         │
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  ClickActionModel(ActionModel):                             │
  │      click: ClickElementAction   ← only this field          │
  │                                                             │
  │  NavigateActionModel(ActionModel):                          │
  │      navigate: NavigateAction    ← only this field          │
  │                                                             │
  │  InputActionModel(ActionModel):                             │
  │      input: InputTextAction      ← only this field          │
  │                                                             │
  │  ScrollActionModel(ActionModel):                            │
  │      scroll: ScrollAction        ← only this field          │
  │  ... (one model per action)                                 │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                    Union them:
                         │
                         ▼
  ActionModel = ClickActionModel | NavigateActionModel | InputActionModel | ...
                         │
                    Wrap in AgentOutput:
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  AgentOutput:                                               │
  │    thinking: str                                            │
  │    evaluation_previous_goal: str                            │
  │    memory: str                                              │
  │    next_goal: str                                           │
  │    action: list[ ActionModel ]   ← 1 to 5 actions           │
  └──────────────────────┬──────────────────────────────────────┘
                         │
              .model_json_schema()
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  JSON Schema (what the LLM actually receives):             │
  │                                                             │
  │  {                                                          │
  │    "properties": {                                          │
  │      "thinking": {"type": "string"},                        │
  │      "evaluation_previous_goal": {"type": "string"},        │
  │      "memory": {"type": "string"},                          │
  │      "next_goal": {"type": "string"},                       │
  │      "action": {                                            │
  │        "type": "array",                                     │
  │        "items": {                                           │
  │          "oneOf": [                                         │
  │            {"properties": {"click": {"properties": {        │
  │              "index": {"type":"integer"},                    │
  │              "coordinate_x": {"type":"integer"}, ...}}}},   │
  │            {"properties": {"navigate": {"properties": {     │
  │              "url": {"type":"string"},                       │
  │              "new_tab": {"type":"boolean"}}}}},             │
  │            {"properties": {"input": {"properties": {        │
  │              "index": {"type":"integer"},                    │
  │              "text": {"type":"string"}, ...}}}},            │
  │            ...                                              │
  │          ]                                                  │
  │        }                                                    │
  │      }                                                      │
  │    }                                                        │
  │  }                                                          │
  └──────────────────────┬──────────────────────────────────────┘
                         │
          Sent to LLM as a forced tool:
                         │
          ┌──────────────┼──────────────────────┐
          │   Anthropic   │   OpenAI             │
          │   tool_use    │   structured_output  │
          │   forced via  │   response_format    │
          │   tool_choice │                      │
          └──────────────┴──────────────────────┘
```


## Diagram 3: LLM Path vs Direct (No-LLM) Path

```
                    ┌──────────────────────┐
                    │     YOUR TASK:       │
                    │  "Click the Submit   │
                    │   button on the form"│
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       WITH LLM (Agent)                  WITHOUT LLM (Direct)
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌──────────────────────┐
   │ Agent sends browser  │          │ You call:            │
   │ state to LLM:        │          │                      │
   │                      │          │ state = await session │
   │ "[1] <h1>Form</h1>   │          │   .get_browser_      │
   │  [2] <input name>    │          │    state_summary()   │
   │  [3] <input email>   │          │                      │
   │  [4] <button>Submit  │          │ map = await session   │
   │       </button>"     │          │   .get_selector_map()│
   └──────────┬──────────┘          └──────────┬───────────┘
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌──────────────────────┐
   │ LLM returns JSON:    │          │ You find the element: │
   │                      │          │                      │
   │ {                    │          │ for idx, el in map:  │
   │   "thinking": "I see │          │   if el.tag == 'button│
   │     submit at [4]",  │          │     and 'Submit' in  │
   │   "action": [{       │          │     el.text:         │
   │     "click": {       │          │       submit_idx = idx│
   │       "index": 4     │          │                      │
   │     }                │          │ # submit_idx = 4     │
   │   }]                 │          │                      │
   │ }                    │          │                      │
   └──────────┬──────────┘          └──────────┬───────────┘
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌──────────────────────┐
   │ Agent.multi_act()    │          │ You call:            │
   │                      │          │                      │
   │ tools.act(           │          │ tools.click(         │
   │   action=ActionModel │          │   index=4,           │
   │     (click={index:4})│          │   browser_session=   │
   │   browser_session=.. │          │     session          │
   │ )                    │          │ )                    │
   └──────────┬──────────┘          └──────────┬───────────┘
              │                                 │
              └────────────┬────────────────────┘
                           │
                    SAME CODE PATH
                           │
                           ▼
              ┌────────────────────────┐
              │  Registry.execute_action│
              │    ("click",           │
              │     params={index: 4}, │
              │     browser_session=.. │
              │    )                   │
              │                        │
              │  Validates params      │
              │  Injects browser_session│
              │  Calls click() func    │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  browser_session       │
              │    .event_bus          │
              │    .dispatch(          │
              │      ClickElementEvent(│
              │        index=4         │
              │      )                 │
              │    )                   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  CDP WebSocket call    │
              │  to Chrome browser     │
              │                        │
              │  DOM.querySelector()   │
              │  Input.dispatchMouse() │
              │                        │
              │  *actual click happens* │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  ActionResult(         │
              │    extracted_content=  │
              │      'Clicked button   │
              │       "Submit"',      │
              │    error=None          │
              │  )                     │
              └────────────────────────┘
```


## Diagram 4: Action Registration (What Happens at Startup)

```
Tools.__init__()
       │
       │  Creates Registry instance
       │
       ▼
  For each action, the @self.registry.action() decorator runs:
       │
       │   Example: registering the "click" action
       │
       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  @self.registry.action('', param_model=ClickElementAction)       │
  │  async def click(params: ClickElementAction,                     │
  │                  browser_session: BrowserSession):                │
  │      ...actual click logic...                                    │
  │                                                                  │
  └──────────────────────────┬───────────────────────────────────────┘
                             │
         decorator inspects the function signature
                             │
                ┌────────────┴────────────────┐
                │                             │
         Action Params               Special Params
         (from param_model)          (injected at runtime)
                │                             │
                ▼                             ▼
  ┌──────────────────────┐      ┌──────────────────────────┐
  │ ClickElementAction:  │      │ browser_session           │
  │   index: int|None    │      │ page_extraction_llm       │
  │   coordinate_x: int? │      │ file_system               │
  │   coordinate_y: int? │      │ sensitive_data             │
  │                      │      │ context                    │
  │ LLM sees these ───►  │      │ cdp_client                │
  │ User passes these ►  │      │                            │
  └──────────────────────┘      │ ◄─── LLM NEVER sees these │
                                │ ◄─── Framework injects     │
                                └──────────────────────────┘
                             │
         decorator creates a normalized wrapper function
         and stores it in the registry:
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────┐
  │  registry.actions["click"] = RegisteredAction(           │
  │    name = "click",                                       │
  │    description = "",                                     │
  │    param_model = ClickElementAction,     ← for schema    │
  │    function = normalized_click_wrapper,  ← for execution │
  │    terminates_sequence = False,                          │
  │    domains = None,                       ← URL filter    │
  │  )                                                       │
  └──────────────────────────────────────────────────────────┘
```


## Diagram 5: Special Params Injection

```
  What the LLM outputs:              What the framework adds:
  ─────────────────────              ───────────────────────

  {"input": {                        browser_session = <BrowserSession>
     "index": 3,                     page_extraction_llm = <ChatOpenAI>
     "text": "<secret>pw</secret>",  sensitive_data = {"pw": "hunter2"}
     "clear": true                   file_system = <FileSystem>
  }}                                 has_sensitive_data = True

        │                                     │
        └──────────────┬──────────────────────┘
                       │
                       ▼
          Registry.execute_action()
                       │
                       ├─ 1. Validate: InputTextAction(index=3, text="<secret>pw</secret>", clear=True)
                       │
                       ├─ 2. Replace secrets: text becomes "hunter2"
                       │
                       ├─ 3. Call: input_func(
                       │         params=InputTextAction(index=3, text="hunter2", clear=True),
                       │         browser_session=<BrowserSession>,
                       │         has_sensitive_data=True,
                       │       )
                       │
                       └─ 4. Return: ActionResult(extracted_content='Typed "***" into index 3')
                                                                     ^^^ masked in logs
```
