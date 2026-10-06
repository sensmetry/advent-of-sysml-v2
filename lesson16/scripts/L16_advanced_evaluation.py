import pathlib
import syside

# Path to our SysML model file
LESSON_DIR = pathlib.Path(__file__).parent.parent
MODEL_FILE_PATH = LESSON_DIR / "models" / "L16_AdvancedEvaluation.sysml" 
# MODEL_FILE_PATH = LESSON_DIR / "models" / "L16_AdvancedEvaluation_WithSanta.sysml" 
CONTEXT = syside.vm.Context(syside.Environment.get_default().lib)


def find_element_by_name(model: syside.Model, name: str) -> syside.Element | None:
    """Search the model for a specific element by name."""

    for element in model.elements(syside.Element, include_subtypes=True):
        if element.name == name:
            return element
    return None


def evaluate_feature(
    feature: syside.Feature, scope: syside.Type
) -> syside.Value | None:
    compiler = syside.Compiler()
    value, compilation_report = compiler.evaluate_feature(
        feature=feature,
        scope=scope,
        context=CONTEXT,
        experimental_quantities=True,
    )
    if compilation_report.fatal:
        print(compilation_report.diagnostics)
        exit(1)
    return value


def main() -> None:
    # Load the SysML model; raises syside.ModelError on any error or warning
    model = syside.load_model([MODEL_FILE_PATH], warnings_as_errors=True)

    # Find the SantaSleigh element in the model
    santa_sleigh = find_element_by_name(model, "SantaSleigh")

    reindeer_count = evaluate_feature(santa_sleigh["reindeerCount"], santa_sleigh)
    total_power = evaluate_feature(santa_sleigh["totalPower"], santa_sleigh)
    average_power = evaluate_feature(santa_sleigh["averagePower"], santa_sleigh)

    print("=======================================================")
    print(f"Total reindeer pulling the sleigh: {reindeer_count}")
    print(f"Total power: {total_power/1000:.2f} kW")
    print(f"Average power per reindeer: {average_power/1000:.2f} kW")

    total_weight = evaluate_feature(santa_sleigh["totalWeight"], santa_sleigh)
    load_per_reindeer = evaluate_feature(santa_sleigh["loadPerReindeer"], santa_sleigh)
    safety_margin = evaluate_feature(santa_sleigh["safetyMargin"], santa_sleigh)

    print("=======================================================")
    print(f"Total weight of SantaSleigh system: {total_weight} kg")
    print(f"Total load pulled per reindeer: {load_per_reindeer:.2f} kg")
    print(f"Remaining safety margin: {safety_margin:.2f} kg")

    flight_mode = evaluate_feature(santa_sleigh["flightMode"], santa_sleigh)
    max_speed = evaluate_feature(santa_sleigh["maxSpeed"], santa_sleigh)

    print("=======================================================")
    print(f"SantaSleigh is operating in: {flight_mode.name} mode")
    print(f"Max speed: {max_speed:.2f} m/s")
    print("=======================================================")

if __name__ == "__main__":
    main()

