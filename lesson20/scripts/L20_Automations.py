import pathlib
import markdown
import syside
from datetime import datetime
import re

## ============================================================================
## CONFIGURATION
## ============================================================================
LESSON_DIR = pathlib.Path(__file__).parent.parent
MODEL_FILE_PATH = LESSON_DIR / "models" / "L20_SantaSleigh.sysml" 
STANDARD_LIBRARY = syside.Environment.get_default().lib
CSS_FILE_PATH = LESSON_DIR / "scripts" / "styles.css"
MARKDOWN_REPORT_PATH = LESSON_DIR / "reports" / "report.md"
HTML_REPORT_PATH = LESSON_DIR / "reports" / "index.html"

## ============================================================================
## SYSML MODEL PARSING
## ============================================================================
class ModelParsing:
    @staticmethod
    def find_element_by_name(model: syside.Model, name: str) -> syside.Element | None:
        """Search the model for a specific element by name."""

        for element in model.elements(syside.Element, include_subtypes=True):
            if element.name == name:
                return element
        return None

    @staticmethod
    def find_inherited_features_by_name(parent: syside.Element, name: str) -> list[syside.Feature]:
        """Search for inherited features by name."""
        result = []
        for feature in parent.inherited_features:
            if feature.name == name:
                result.append(feature)
        return result

    @staticmethod
    def find_inherited_features_by_type(parent: syside.Element, searchType: syside.Type) -> list[syside.Feature]:
        """Search for inherited features by type."""
        result = []
        for feature in parent.inherited_features:
            if type(feature) is searchType:
                result.append(feature)
        return result

    @staticmethod
    def find_owned_elements_by_name(parent: syside.Element, name: str) -> syside.Element | None:
        """Search for a specific owned element by name."""

        for element in parent.owned_elements:
            if element.name == name:
                return element
        return None

    @staticmethod
    def evaluate_feature(
        feature: syside.Feature, scope: syside.Type
    ) -> syside.Value | None:
        """Evaluate a feature within a given scope and return its computed value."""
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

## ============================================================================
## REPORT GENERATION
## ============================================================================
class ReportGenerator:
    def generate_report(model : syside.Model, instance_name : str, title : str) -> list[str]:
        """Generate a complete report for a delivery instance."""
        # Find the delivery_run instance
        delivery_run = ModelParsing.find_element_by_name(model, instance_name)

        # Extract section data
        summary_data = DataExtraction.extract_summary_data(delivery_run)
        team_roster_data = DataExtraction.extract_team_data(delivery_run)
        sleigh_comp_data = DataExtraction.extract_sleigh_comp_data(delivery_run)
        gift_bag_data = DataExtraction.extract_gift_data(delivery_run)

        report = []

        # Header
        report.append(f"# {title}")
        report.append(f"\n**Delivery:** {instance_name}")
        report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append("\n---\n")

        # Append section content
        report.extend(ReportGenerator.generate_summary_table(summary_data))
        report.extend(ReportGenerator.generate_team_roster_table(team_roster_data))
        report.extend(ReportGenerator.generate_sleigh_composition_table(sleigh_comp_data))
        report.extend(ReportGenerator.generate_cargo_summary_table(gift_bag_data))
        report.extend(ReportGenerator.generate_cargo_breakdown(gift_bag_data))

        return report

    def generate_summary_table(data):
        """Generate markdown table for executive summary."""
        table = []

        table.append("## Executive Summary\n")
        table.append("| Metric | Value |")
        table.append("|--------|-------|")
        table.append(f"| Total gifts | {data['total_gifts']} items |")
        table.append(f"| Number of bags | {data['number_of_bags']} routes |")
        table.append(f"| Total cargo weight | {data['cargo_weight']:.2f} |")
        table.append(f"| Sleigh total weight | {data['sleight_weight']:.2f} |")
        table.append(f"| Load per reindeer | {data['load_per_reindeer']:.2f} |")
        table.append(f"| Safety margin | {data['safety_margin']:.2f} |")
        table.append(f"| Flight mode | {data['flight_mode']} |")
        table.append(f"| Max allowed speed | {data['max_speed']:.2f} |")
        table.append("\n---\n")

        return table

    def generate_team_roster_table(data):
        """Generate markdown table for team roster."""

        table = []
        table.append("## Reindeer Team Roster\n")

        no = 1
        for reindeer in data:
            table.append(f"### Crew #{no}: {reindeer['name']}\n")
            table.append(f"{reindeer['description_en']}\n")
            table.append(f"*{reindeer['description_fr']}*\n")

            table.append("| Attribute | Value |")
            table.append("|-----------|-------|")
            table.append(f"| Nose Color | {reindeer['nose_color']} |")
            table.append(f"| Weight | {reindeer['weight']:.2f} kg |")
            table.append(f"| Power | {reindeer['power']:.2f} W |")
            table.append(f"| Energy Level | {reindeer['energy_level']:.2f} W |")
            table.append("\n---\n")
            no += 1

        return table

    def generate_sleigh_composition_table(data):
        """Generate markdown table for sleigh composition."""

        table = []
        table.append("## Sleigh Composition\n")
        table.append("| Part | Description (EN) | Description (FR) |")
        table.append("|------|------------------|------------------|")

        for part in data:
            table.append(
                f"| {part['name']} | "
                f"{part['description_en']} | "
                f"{part['description_fr']} |"
            )

        table.append("\n")
        return table

    def generate_cargo_summary_table(data):
        """Generate summary table of all gift bags."""

        table = []
        table.append("## Cargo Summary\n")
        table.append("| Route | Gift Count | Total Weight |")
        table.append("|-------|------------|-------------|")

        for bag in data:
            table.append(
                f"| {bag['name']} | "
                f"{bag['gift_count']} gifts | "
                f"{bag['weight']:.2f} kg |"
            )

        table.append("\n")
        return table

    def generate_cargo_breakdown(data):
        """Generate detailed breakdown with individual gift tables per bag."""

        table = []
        table.append("## Cargo Breakdown by Route\n")

        for bag in data:
            table.append(f"### {bag['name']}\n")
            table.append(f"**Items**: {bag['gift_count']} gifts\n")
            table.append(f"**Weight**: {bag['weight']:.2f} kg\n")

            # Individual gifts table
            table.append("| Gift | Weight |")
            table.append("|------|--------|")

            for gift in bag['gifts']:
                table.append(f"| {gift['name']} | {gift['weight']:.2f} kg |")

            table.append("\n---\n")

        return table
    
    def convert_to_html(md_content: str, title : str, footer : str) -> str:
        """Convert markdown report to HTML with embedded styling."""
        html_body = markdown.markdown(md_content, extensions=['tables'])

        # Split content to separate header from rest
        parts = re.split(r'<hr\s*/?>', html_body, maxsplit=1)
        header_content = parts[0] if len(parts) > 0 else ""
        main_content = parts[1] if len(parts) > 1 else html_body

        # Read CSS file and embed it
        css_path = CSS_FILE_PATH
        css_content = css_path.read_text(encoding='utf-8')

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
{css_content}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
{header_content}
        </div>
        <div class="content">
{main_content}
        </div>
        <div class="footer">
            {footer}
        </div>
    </div>
</body>
</html>"""

## ============================================================================
## DATA EXTRACTION
## ============================================================================
class DataExtraction:
    def extract_team_data(delivery_run : syside.Element) -> dict:
        """Extract reindeer team roster data."""
        reindeer_features = ModelParsing.find_inherited_features_by_name(delivery_run, "reindeer")
        assert(len(reindeer_features) == 1)
        team_roster = ModelParsing.evaluate_feature(reindeer_features[0], delivery_run)

        team_roster_data =[]
        for team_member in team_roster:
            desc_en = ModelParsing.find_owned_elements_by_name(team_member, "DescriptionEN")
            desc_fr = ModelParsing.find_owned_elements_by_name(team_member, "DescriptionFR")

            nose_color = ModelParsing.find_inherited_features_by_name(team_member, 'noseColor')[0]
            weight = ModelParsing.find_inherited_features_by_name(team_member, 'weight')[0]
            power = ModelParsing.find_inherited_features_by_name(team_member, 'power')[0]
            energy_level = ModelParsing.find_inherited_features_by_name(team_member, 'energyLevel')[0]

            team_roster_data.append({
                'name' : team_member.name,
                'description_en' : desc_en.body if hasattr(desc_en, 'body') else "<MISSING EN DESCRIPTION>",
                'description_fr' : desc_fr.body if hasattr(desc_fr, 'body') else "<MISSING FR DESCRIPTION>",
                'nose_color' : ModelParsing.evaluate_feature(nose_color, team_member).name,
                'weight' : ModelParsing.evaluate_feature(weight, team_member),
                'power' : ModelParsing.evaluate_feature(power, team_member),
                'energy_level' : ModelParsing.evaluate_feature(energy_level, team_member),
            })

        return team_roster_data

    def extract_sleigh_comp_data(delivery_run : syside.Element) -> dict:
        """Extract sleigh composition and part data."""
        sleigh_parts = ModelParsing.find_inherited_features_by_type(delivery_run, syside.PartUsage)
        # Filter out reference and SysML library parts
        sleigh_parts = [part for part in sleigh_parts if part.is_reference is False and part.is_library_element is False]

        sleigh_comp_data = []
        for sleigh_part in sleigh_parts:
            desc_en = ModelParsing.find_owned_elements_by_name(sleigh_part, "DescriptionEN")
            desc_fr = ModelParsing.find_owned_elements_by_name(sleigh_part, "DescriptionFR")
            sleigh_comp_data.append({
                'name' : sleigh_part.name,
                'description_en' : desc_en.body if hasattr(desc_en, 'body') else "<MISSING EN DESCRIPTION>",
                'description_fr' : desc_fr.body if hasattr(desc_fr, 'body') else "<MISSING FR DESCRIPTION>",
            })

        return sleigh_comp_data

    def extract_gift_data(delivery_run : syside.Element) -> dict:
        """Extract cargo gift bag and individual gift data."""
        cargo_payload = delivery_run['cargoBay']['payload']
        gift_bags = ModelParsing.evaluate_feature(cargo_payload, delivery_run)

        gift_bag_data =[]
        for gift_bag in gift_bags:
            gift_count = ModelParsing.find_inherited_features_by_name(gift_bag, 'giftCount')[0]
            gift_weight = ModelParsing.find_inherited_features_by_name(gift_bag, 'totalWeight')[0]

            gifts = ModelParsing.evaluate_feature(gift_bag['gifts'], gift_bag)
            gift_data = []
            for gift in gifts:
                weight = ModelParsing.find_inherited_features_by_name(gift, 'weight')[0]
                gift_data.append({
                    'name' : gift.name,
                    'weight' : ModelParsing.evaluate_feature(weight, gift)
                })

            gift_bag_data.append({
                'name' : gift_bag.name,
                'gift_count' : ModelParsing.evaluate_feature(gift_count, gift_bag),
                'weight' : ModelParsing.evaluate_feature(gift_weight, gift_bag),
                'gifts' : gift_data
            })

        return gift_bag_data

    def extract_summary_data(delivery_run : syside.Element) -> dict:
        """Extract executive summary metrics."""
        gift_count = ModelParsing.find_inherited_features_by_name(delivery_run['cargoBay'], 'giftCount')[0]
        bag_count = ModelParsing.find_inherited_features_by_name(delivery_run['cargoBay'], 'bagCount')[0]
        bag_weight = ModelParsing.find_inherited_features_by_name(delivery_run['cargoBay'], 'weight')[0]
        sleight_weight = ModelParsing.find_inherited_features_by_name(delivery_run, 'totalWeight')[0]
        load_per_reindeer = ModelParsing.find_inherited_features_by_name(delivery_run, 'loadPerReindeer')[0]
        safety_margin = ModelParsing.find_inherited_features_by_name(delivery_run, 'safetyMargin')[0]
        flight_mode = ModelParsing.find_inherited_features_by_name(delivery_run, 'flightMode')[0]
        max_speed = ModelParsing.find_inherited_features_by_name(delivery_run, 'maxSpeed')[0]

        summary_data = {
            'total_gifts': ModelParsing.evaluate_feature(gift_count, delivery_run['cargoBay']),
            'number_of_bags' : ModelParsing.evaluate_feature(bag_count, delivery_run['cargoBay']),
            'cargo_weight' : ModelParsing.evaluate_feature(bag_weight, delivery_run['cargoBay']),
            'sleight_weight' : ModelParsing.evaluate_feature(sleight_weight, delivery_run),
            'load_per_reindeer' : ModelParsing.evaluate_feature(load_per_reindeer, delivery_run),
            'safety_margin' : ModelParsing.evaluate_feature(safety_margin, delivery_run),
            'flight_mode' : ModelParsing.evaluate_feature(flight_mode, delivery_run).name,
            'max_speed' : ModelParsing.evaluate_feature(max_speed, delivery_run),
        }

        return summary_data


def main() -> None:
    # Load SysML model and get diagnostics (errors/warnings)
    (model, diagnostics) = syside.load_model([MODEL_FILE_PATH])

    # Make sure the model contains no errors before proceeding
    assert not diagnostics.contains_errors(warnings_as_errors=True)

    # Generate the report
    title = "Santa's Sleigh 2025 - Bill of Materials"
    footer = "Merry Christmas and Happy Holidays!"
    report_lines = ReportGenerator.generate_report(model, "deliveryRun1382", title)

    # Write to markdown file
    output_path = MARKDOWN_REPORT_PATH
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text("\n".join(report_lines), encoding='utf-8')

    # Write HTML
    html_content = ReportGenerator.convert_to_html("\n".join(report_lines), title, footer)
    html_path = HTML_REPORT_PATH
    html_path.write_text(html_content, encoding='utf-8')

    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    main()

