                               package L03_SantaWorkshop_Challenge {
private import ScalarValues::*;

part def SantaWorkshop {
// Work areas of the workshop
part assemblyLine {doc /* Where toys are assembled. */ part supervisor = headElf;}part qualityControl { doc /* Quality checking station. */ }part packagingArea { doc /* Gift wrapping and packaging. */ }

// Elves working in the workshop
part headElf;

part albert {  attribute skillType : SkillType = SkillType::assembly;attribute skillLevel : Real := 50.0; }
        part betty 
{attribute skillType : SkillType = SkillType::assembly;
                attribute skillLevel : Real := 100.0; }
   part charlie {  attribute skillType : SkillType = SkillType::quality;attribute skillLevel : Real := 100.0; }
        part diana {  attribute skillType : SkillType = SkillType::quality;
          attribute skillLevel : Real := 100.0; }
 part edward {  attribute skillType : SkillType = SkillType::packaging;
                          attribute skillLevel : Real := 100.0; }
                                                                                                           part fiona { attribute skillType : SkillType = SkillType::packaging;
                         attribute skillLevel : Real := 100.0; }
        part george {  attribute skillType : SkillType = SkillType::assembly;
attribute skillLevel : Real := 150.0; }
        part helen { 
    attribute skillType : SkillType = SkillType::quality;
attribute skillLevel : Real := 150.0; }}
enum def SkillType {
enum assembly; enum quality; enum packaging;}}