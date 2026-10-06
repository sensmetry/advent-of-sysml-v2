package L09_SantaWorkshop {
        private import ScalarValues::*;
        
        // Install Sysand package for PLEML from Sysand and uncomment line below.
        // private import PLEML::*;

        part def SantaWorkshop {
                // Work areas of the workshop
                part assemblyLine {
                        doc
                        /* Where toys are assembled. */
                        part supervisor = headElf;
                }
                part qualityControl {
                        doc
                        /* Quality checking station. */
                }
                part packagingArea {
                        doc
                        /* Gift wrapping and packaging. */
                }

                // Elves working in the workshop
                part headElf : Elf;

                part albert : Elf {
                        attribute :>> skillType := SkillType::assembly;
                        attribute :>> skillLevel := 50.0;
                }
                part betty : Elf {
                        attribute :>> skillType := SkillType::assembly;
                        attribute :>> skillLevel := 100.0;
                }
                part charlie : Elf {
                        attribute :>> skillType := SkillType::quality;
                        attribute :>> skillLevel := 100.0;
                }
                part diana : Elf {
                        attribute :>> skillType := SkillType::quality;
                        attribute :>> skillLevel := 100.0;
                }
                part edward : Elf {
                        attribute :>> skillType := SkillType::packaging;
                        attribute :>> skillLevel := 100.0;
                }
                part fiona : Elf {
                        attribute :>> skillType := SkillType::packaging;
                        attribute :>> skillLevel := 100.0;
                }
                part george : Elf {
                        attribute :>> skillType := SkillType::assembly;
                        attribute :>> skillLevel := 150.0;
                }
                part helen : Elf {
                        attribute :>> skillType := SkillType::quality;
                        attribute :>> skillLevel := 150.0;
                }
        }

        part def Elf {
                attribute skillType : SkillType default := SkillType::TBD;
                attribute skillLevel : Real default := 0.0;
        }

        enum def SkillType {
                enum TBD;
                enum assembly;
                enum quality;
                enum packaging;
        }
}
