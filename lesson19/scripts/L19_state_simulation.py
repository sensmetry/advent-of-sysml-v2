import pathlib
import syside
import re
import csv
from sismic.io import import_from_yaml
from sismic.interpreter import Interpreter
from sismic.model import Event
from graphviz import Source
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## ============================================================================
## CONFIGURATION
## ============================================================================
LESSON_DIR = pathlib.Path(__file__).parent.parent
MODEL_FILE_PATH = LESSON_DIR / "models" / "L19_State_Simulation.sysml"
STANDARD_LIBRARY = syside.Environment.get_default().lib

SIM_SCENARIO_FILE_PATH = LESSON_DIR / "scripts" / "sim_scenario_nominal.csv" # Set to None for interactive mode
SIM_STEP_DELAY = 0.1 # Delay between scenario steps in seconds

## ============================================================================
## SYSML MODEL EXTRACTION
## ============================================================================
class ModelExtraction:
    @staticmethod
    def load_model_from_file(path: pathlib.Path) -> syside.Model | None:
        """Load SysML model from configured path"""
        (model, diagnostics) = syside.load_model([path])

        if diagnostics.contains_errors():
            print(diagnostics)
            return None

        return model

    @staticmethod
    def find_element_by_name(model: syside.Model, name: str) -> syside.Element | None:
        """Search the model for a specific element by name."""
        for element in model.elements(syside.Element, include_subtypes=True):
            if element.name == name:
                return element
        return None

    @staticmethod
    def find_owned_elements_by_type(parent: syside.Element, searchType: syside.Type) -> list[syside.Element]:
        """Search the model for a specific elements by type."""
        result = []
        for element in parent.owned_elements:
            if type(element) is searchType:
                result.append(element)
        return result

    @staticmethod
    def evaluate_feature(feature: syside.Feature, scope: syside.Type) -> syside.Value | None:
        """Evaluate a SysML feature and return its computed value."""
        compiler = syside.Compiler()
        value, compilation_report = compiler.evaluate_feature(
            feature=feature,
            scope=scope,
            stdlib=STANDARD_LIBRARY,
            experimental_quantities=True,
        )
        if compilation_report.fatal:
            print(compilation_report.diagnostics)
            exit(1)
        return value

    @staticmethod
    def build_state_object(state: syside.StateUsage) -> dict:
        """Build a dictionary representation of a state including substates."""
        substates = [x for x in state.owned_elements if type(x) is syside.StateUsage]

        initial_substate = None
        if substates:
            successions = ModelExtraction.find_owned_elements_by_type(state, syside.SuccessionAsUsage)
            if successions:
                initial_substate = successions[0].targets[0].name

        return {
            "name": state.name,
            "type": state.is_parallel,
            "states": [ModelExtraction.build_state_object(substate) for substate in substates],
            "initial": initial_substate,
            "transitions": [],  # populated separately
        }

    @staticmethod
    def build_transition_object(transition: syside.TransitionUsage) -> dict:
        """Build a dictionary representation of a transition with event and guard."""
        trigger_event = transition.trigger_action.payload_parameter.feature_target.heritage[0][1]
        
        guard_str = syside.pprint(transition.guard_expression).strip() if transition.guard_expression is not None else ""
        # Remove units ([kg], [m/s], etc.) from expression
        guard_str = re.sub(r"\s*\[[\w\s/]+\]", "", guard_str)

        return {
            "source": transition.source.name,
            "target": transition.target.name,
            "event": trigger_event.name,
            "guard": guard_str,
        }

    @staticmethod
    def collect_all_transitions(model: syside.Model) -> list[dict]:
        """Collect all transitions from the SysML model."""
        transitions = []
        for transition in model.elements(syside.TransitionUsage):
            transitions.append(ModelExtraction.build_transition_object(transition))
        return transitions

    @staticmethod
    def inject_transitions(states : dict, all_transitions: list[dict]) -> None:
        """Recursively assign transitions to their source states."""
        for state in states:
            state["transitions"] = [t for t in all_transitions if t["source"] == state["name"]]

            # Recurse into substates
            if state["states"]:
                ModelExtraction.inject_transitions(state["states"], all_transitions)

    @staticmethod
    def extract_state_machine(model: syside.Model, state_machine_name: str):
        """Extract state machine structure from SysML model using Syside Automator"""
        state_machine_element = ModelExtraction.find_element_by_name(model, state_machine_name)

        states = ModelExtraction.build_state_object(state_machine_element)
        all_transitions = ModelExtraction.collect_all_transitions(model)
        ModelExtraction.inject_transitions(states["states"], all_transitions)

        # Find entry point
        successions = ModelExtraction.find_owned_elements_by_type(
            state_machine_element, syside.SuccessionAsUsage
        )
        assert len(successions) == 1
        initial_state = successions[0].targets[0]

        # Extract simulation parameters
        attributes = ModelExtraction.find_owned_elements_by_type(
            state_machine_element, syside.AttributeUsage
        )
        parameters = {}
        for attribute in attributes:
            parameters[attribute.name] = ModelExtraction.evaluate_feature(
                attribute, state_machine_element
            )

        # Extract event names from transitions
        event_names = list(set(t["event"] for t in all_transitions if t.get("event")))

        return {
            "name": state_machine_name,
            "states": states["states"],
            "transitions": states["transitions"],
            "initial_state": initial_state.name,
            "parameters": parameters,
            "event_names": event_names,
        }

## ============================================================================
## STATE MACHINE GENERATION
## ============================================================================
class StateMachineGen:
    @staticmethod
    def generate_state_yaml(state, indent_level):
        """Recursively generate YAML for a state and its children"""
        indent = "  " * indent_level
        lines = [f"{indent}- name: {state['name']}"]

        if state["transitions"]:
            lines.append(f"{indent}  transitions:")
            for trans in state["transitions"]:
                lines.append(f"{indent}    - target: {trans['target']}")
                lines.append(f"{indent}      event: {trans['event']}")
                if trans.get("guard"):
                    lines.append(f"{indent}      guard: {trans['guard']}")

        if state["states"]:
            if state.get("initial"):
                lines.append(f"{indent}  initial: {state['initial']}")

            if state.get("type"):
                lines.append(f"{indent}  parallel states:")
            else:
                lines.append(f"{indent}  states:")
            for substate in state["states"]:
                lines.extend(
                    StateMachineGen.generate_state_yaml(substate, indent_level + 2)
                )

        return lines

    @staticmethod
    def generate_sismic_yaml(state_machine_data):
        """Generate sismic YAML from extracted state machine data"""
        param_lines = []
        for var_name, var_value in state_machine_data["parameters"].items():
            param_lines.append(f"    {var_name} = {var_value}")

        param_str = "\n".join(param_lines) if param_lines else "    pass"

        states_lines = []
        for state in state_machine_data["states"]:
            states_lines.extend(StateMachineGen.generate_state_yaml(state, 3))

        states_str = "\n".join(states_lines)

        yaml_template = \
f"""statechart:
  name: {state_machine_data["name"]}
  preamble: |
{param_str}
  root state:
    name: root
    initial: {state_machine_data["initial_state"]}
    states:
{states_str}
"""

        return yaml_template

## ============================================================================
## VISUALIZATION
## ============================================================================
class Visualization:
    @staticmethod
    def statechart_to_dot(statechart, active_states, last_transition=None):
        """Convert sismic statechart to DOT format"""
        lines = [
            "digraph Diagram {",
            "  rankdir=TB;",
            "  node [shape=box, style=rounded];",
            "  compound=true;",
        ]

        # Build hierarchy map
        state_children = {}
        for state_name in statechart.states:
            if state_name == "root":
                continue
            children = statechart.children_for(state_name)
            if children:
                state_children[state_name] = children

        # Collect all states
        all_states = [s for s in statechart.states if s != "root"]

        # Recursive function to render states
        def render_state(state_name, indent_level=1):
            indent = "  " * indent_level
            state_lines = []

            if state_name in state_children:
                # This is a composite state - create cluster
                # Highlight cluster if any of its descendants are active
                cluster_style = 'rounded'
                if state_name in active_states:
                    cluster_style = 'rounded,filled'
                    state_lines.append(f'{indent}subgraph cluster_{state_name} {{')
                    state_lines.append(f'{indent}  label="{state_name}";')
                    state_lines.append(f'{indent}  style="{cluster_style}";')
                    state_lines.append(f'{indent}  fillcolor=lightblue;')
                else:
                    state_lines.append(f'{indent}subgraph cluster_{state_name} {{')
                    state_lines.append(f'{indent}  label="{state_name}";')
                    state_lines.append(f'{indent}  style="{cluster_style}";')

                # Render children
                for child in state_children[state_name]:
                    state_lines.extend(render_state(child, indent_level + 1))

                state_lines.append(f'{indent}}}')
            else:
                # This is a leaf state - create node
                if state_name in active_states:
                    state_lines.append(f'{indent}{state_name} [style="rounded,filled", fillcolor=yellow];')
                else:
                    state_lines.append(f'{indent}{state_name};')

            return state_lines

        # Render all top-level states
        root_children = statechart.children_for("root")
        for state in root_children:
            lines.extend(render_state(state, 1))

        # Add transitions
        for state in all_states:
            for transition in statechart.transitions_from(state):
                event = transition.event or ""
                guard = f"[{transition.guard}]" if transition.guard else ""
                label = f"{event} {guard}".strip()

                # Get target state - if target is composite, point to its initial substate
                target = transition.target
                if target in state_children:
                    # Target is a composite state - find its initial substate
                    initial = statechart.state_for(target).initial
                    if initial:
                        target = initial

                # Get source states - if source is composite, draw from each leaf substate
                source_states = [state]
                if state in state_children:
                    # Source is a composite state - draw from all leaf descendants
                    def get_leaf_descendants(s):
                        if s in state_children:
                            leaves = []
                            for child in state_children[s]:
                                leaves.extend(get_leaf_descendants(child))
                            return leaves
                        else:
                            return [s]
                    source_states = get_leaf_descendants(state)

                # Draw transition from each source state
                for source in source_states:
                    if (
                        last_transition
                        and last_transition["source"] == source
                        and last_transition["target"] == target
                    ):
                        # Highlight last transition
                        lines.append(f'  {source} -> {target} [label="{label}", color=red, penwidth=2.5];')
                    else:
                        lines.append(f'  {source} -> {target} [label="{label}"];')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def render_state_diagram(
        statechart, active_states, output_name, last_transition=None
    ):
        """Render the state diagram"""
        dot_string = Visualization.statechart_to_dot(statechart, active_states, last_transition)
        graph = Source(dot_string)
        output_path = graph.render(output_name, view=False, cleanup=True, format="png")
        return output_path

    @staticmethod
    def plot_timeseries(axes, history, param_names):
        """Plot state and parameter timeseries on given axes"""
        if not history:
            return

        times = [h['time'] for h in history]
        states = [h['state'] for h in history]

        # Get unique states for color mapping
        unique_states = []
        for s in states:
            if s not in unique_states:
                unique_states.append(s)
        state_to_num = {state: i for i, state in enumerate(unique_states)}
        state_nums = [state_to_num[s] for s in states]

        # Plot state changes
        ax_state = axes[0]
        ax_state.clear()
        ax_state.step(times, state_nums, where='post', linewidth=2)
        ax_state.set_ylabel('State', fontweight='bold')
        ax_state.set_yticks(range(len(unique_states)))
        ax_state.set_yticklabels(unique_states)
        ax_state.grid(True, alpha=0.3)
        ax_state.set_xlim(times[0], times[-1])

        # Plot parameters
        for idx, param_name in enumerate(param_names):
            ax = axes[idx + 1]
            ax.clear()
            param_values = [h['params'][param_name] for h in history]
            ax.plot(times, param_values, linewidth=2, marker='o', markersize=3)
            ax.set_ylabel(param_name, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(times[0], times[-1])

        # Only show x-label on bottom plot
        axes[-1].set_xlabel('Time', fontweight='bold')

        # Hide x-labels for all but bottom plot
        for ax in axes[:-1]:
            ax.set_xticklabels([])

## ============================================================================
## SIMULATION
## ============================================================================
class Simulation:
    @staticmethod
    def step_simulation(
        interpreter,
        context_values: dict,
        event_name: str,
        prev_state: str = None,
    ):
        """Execute a single simulation step with given context values"""
        for var_name, value in context_values.items():
            interpreter.context[var_name] = value

        interpreter.queue(Event(event_name))
        interpreter.execute()

        active_states = interpreter.configuration
        current_state = None
        for state in active_states:
            if state != "root":
                children = interpreter.statechart.children_for(state)
                if not any(child in active_states for child in children):
                    current_state = state
                    break
        if not current_state:
            current_state = "unknown"

        last_transition = None
        if prev_state and prev_state != current_state:
            last_transition = {"source": prev_state, "target": current_state}

        return {
            "current_state": current_state,
            "active_states": active_states,
            "last_transition": last_transition,
        }

    @staticmethod
    def run_interactive(interpreter, statechart, output_name, allowed_event_names):
        """Run interactive simulation with user input"""
        print("\nStarting simulation...\n\nType 'quit' or 'q' to exit.")

        # Set up visualization
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis("off")

        # Initial state (find leaf state)
        active_states = interpreter.configuration
        current_state = None
        for state in active_states:
            if state != "root":
                children = statechart.children_for(state)
                if not any(child in active_states for child in children):
                    current_state = state
                    break
        if not current_state:
            current_state = "init"

        img_path = Visualization.render_state_diagram(statechart, active_states, output_name)
        img = mpimg.imread(img_path)
        img_display = ax.imshow(img)

        # Info box with state and context
        context_lines = [f"{k}: {v:.1f}" for k, v in interpreter.context.items()]
        info_text = f"State: {current_state}\n" + "\n".join(context_lines)

        ax.set_title(statechart.name, fontsize=14, fontweight='bold')
        text_box = ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()
        plt.pause(0.5)

        prev_state = current_state

        while True:
            try:
                # Get input for all context variables
                event_name = input(f"Allowed events: {allowed_event_names}\nEnter event name: ").strip()

                context_values = {}
                for var_name in interpreter.context.keys():
                    user_input = input(f" - {var_name}: ").strip()

                    if user_input.lower() in ["quit", "q", "exit"]:
                        print("Exiting simulation...")
                        plt.close('all')
                        return

                    context_values[var_name] = float(user_input)


                # Execute simulation step
                result = Simulation.step_simulation(
                    interpreter,
                    context_values,
                    event_name if event_name in allowed_event_names else None,
                    prev_state,
                )

                current_state = result["current_state"]
                active_states = result["active_states"]
                last_transition = result["last_transition"]

                if last_transition:
                    print(f"  Transition: {last_transition['source']} -> {last_transition['target']}")

                # Update visualization
                img_path = Visualization.render_state_diagram(statechart, active_states, output_name, last_transition)
                img = mpimg.imread(img_path)

                if img_display is None:
                    img_display = ax.imshow(img)
                else:
                    img_display.set_data(img)

                # Update info box with state and context
                context_lines = [f"{k}: {v:.1f}" for k, v in interpreter.context.items()]
                info_text = f"State: {current_state}\n" + "\n".join(context_lines)
                text_box.set_text(info_text)

                ax.axis("off")
                fig.canvas.draw_idle()
                plt.pause(0.1)

                print(f"  -> State: {current_state}")

                prev_state = current_state

            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nExiting simulation...")
                break

        plt.close('all')

    @staticmethod
    def run_scenario(
        interpreter,
        statechart,
        output_name,
        scenario_file: pathlib.Path,
        step_delay: float = 1.0,
    ):
        """Run automated simulation from CSV scenario file"""
        print(f"\nRunning scenario from {scenario_file.name}...")

        # Read and parse CSV
        with open(scenario_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            raw_steps = list(reader)

        if not raw_steps:
            print("Error: Scenario file is empty")
            return

        # Parse and sort by time
        header = list(raw_steps[0].keys())
        param_cols = header[2:]  # Skip time and event

        scenario_keyframes = []
        for step in raw_steps:
            scenario_keyframes.append({
                'time': int(step['time']),
                'event': step['event'].strip(),
                'params': {param: float(step[param]) for param in param_cols}
            })
        scenario_keyframes.sort(key=lambda x: x['time'])

        start_time = scenario_keyframes[0]['time']
        end_time = scenario_keyframes[-1]['time']

        print(f"Time steps: {start_time} to {end_time}")
        print(f"Parameters: {', '.join(param_cols)}\n")

        # Set up visualization
        plt.ion()
        num_plots = 1 + len(param_cols)  # State + parameters
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.35)

        # Top: State diagram
        ax_diagram = fig.add_subplot(gs[0])
        ax_diagram.axis("off")

        # Bottom: Timeseries plots
        gs_bottom = gs[1].subgridspec(num_plots, 1, hspace=0.5)
        ax_timeseries = [fig.add_subplot(gs_bottom[i]) for i in range(num_plots)]

        # Initial state
        active_states = interpreter.configuration
        current_state = None
        for state in active_states:
            if state != "root":
                children = statechart.children_for(state)
                if not any(child in active_states for child in children):
                    current_state = state
                    break
        if not current_state:
            current_state = "init"

        img_path = Visualization.render_state_diagram(statechart, active_states, output_name)
        img = mpimg.imread(img_path)
        img_display = ax_diagram.imshow(img)

        # Info box on diagram
        context_lines = [f"{k}: {v:.1f}" for k, v in interpreter.context.items()]
        info_text = f"Time: 0\nState: {current_state}\n" + "\n".join(context_lines)

        ax_diagram.set_title(statechart.name, fontsize=14, fontweight='bold')
        text_box = ax_diagram.text(0.02, 0.98, info_text, transform=ax_diagram.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.show()
        plt.pause(step_delay)

        prev_state = current_state
        history = []

        # Execute scenario with HOLD_LAST_VAL for gaps
        keyframe_idx = 0
        last_event = scenario_keyframes[0]['event']
        last_params = scenario_keyframes[0]['params'].copy()

        for current_time in range(start_time, end_time + 1):
            # Update to new keyframe if we've reached it
            while keyframe_idx < len(scenario_keyframes) and scenario_keyframes[keyframe_idx]['time'] == current_time:
                last_event = scenario_keyframes[keyframe_idx]['event']
                last_params = scenario_keyframes[keyframe_idx]['params'].copy()
                keyframe_idx += 1

            print(f"Time: {current_time} | {last_event} | {last_params}")

            # Execute simulation step
            result = Simulation.step_simulation(
                interpreter,
                last_params,
                last_event if last_event != 'None' else None,
                prev_state,
            )

            current_state = result["current_state"]
            active_states = result["active_states"]
            last_transition = result["last_transition"]

            # Record history
            history.append({
                'time': current_time,
                'state': current_state,
                'params': last_params.copy()
            })

            if last_transition:
                print(f"  Transition: {last_transition['source']} -> {last_transition['target']}")

            # Update state diagram
            img_path = Visualization.render_state_diagram(
                statechart, active_states, output_name, last_transition
            )
            img = mpimg.imread(img_path)
            img_display.set_data(img)

            # Update info box
            context_lines = [f"{k}: {v:.1f}" for k, v in interpreter.context.items()]
            info_text = f"Time: {current_time}\nState: {current_state}\n" + "\n".join(context_lines)
            text_box.set_text(info_text)

            # Update timeseries plots
            Visualization.plot_timeseries(ax_timeseries, history, param_cols)

            ax_diagram.axis("off")
            fig.canvas.draw_idle()
            plt.pause(step_delay)

            print(f"  -> State: {current_state}\n")

            prev_state = current_state

        print("Scenario complete!")
        plt.ioff()
        plt.show()


def main():
    """
    SysML v2 State Machine Simulation using Syside Automator
    Extracts state machine from SysML model and simulates with sismic
    """
    # Configuration
    state_machine_name = "SleighFlightMonitor"

    # Load SysML model
    model = ModelExtraction.load_model_from_file(MODEL_FILE_PATH)

    # Extract state machine structure
    state_machine_data = ModelExtraction.extract_state_machine(
        model, state_machine_name
    )

    # Generate output filename from state machine name
    output_name = state_machine_name.lower().replace(" ", "_")

    # Generate sismic YAML
    print("\nGenerating sismic YAML...")
    yaml_string = StateMachineGen.generate_sismic_yaml(state_machine_data)
    print(yaml_string)

    # Load statechart and create interpreter
    print("\nLoading statechart...")
    statechart = import_from_yaml(yaml_string)
    interpreter = Interpreter(statechart)
    interpreter.execute()

    # Run simulation (interactive or scenario mode)
    if SIM_SCENARIO_FILE_PATH:
        Simulation.run_scenario(interpreter, statechart, output_name, SIM_SCENARIO_FILE_PATH, SIM_STEP_DELAY)
    else:
        Simulation.run_interactive(interpreter, statechart, output_name, state_machine_data['event_names'])


if __name__ == "__main__":
    main()
