package L02_SantaSleigh {
    private import ScalarValues::*;

    part def SantaSleigh {
        // Components of the sleigh
        part cockpit {
            doc /* This is where Santa sits. */
            ref item driver = santa;
        }
        part cargoBay  { doc /* Place for the GiftBag. */  }
        part dashboard { doc /* Naughty-Nice dashboard. */ }

        // Entities related to the sleigh
        ref item santa;
        abstract ref part reindeer : Reindeer [9];
       
        ref part dasher  subsets reindeer { constant redefines noseColor = Color::orange; }
        ref part dancer  subsets reindeer { constant redefines noseColor = Color::yellow; }
        ref part prancer subsets reindeer { constant redefines noseColor = Color::green;  }
        ref part vixen   subsets reindeer { constant redefines noseColor = Color::blue;   }
        ref part comet   subsets reindeer { constant redefines noseColor = Color::indigo; }
        ref part cupid   subsets reindeer { constant redefines noseColor = Color::violet; }
        ref part donner  subsets reindeer { constant redefines noseColor = Color::silver; }
        ref part blitzen subsets reindeer { constant redefines noseColor = Color::gold;   }
        ref part rudolph subsets reindeer { constant redefines noseColor = Color::red;    
                                            attribute redefines energyLevel := 200.0;      }
    }

    part def Reindeer {
        constant attribute noseColor : Color;
        attribute energyLevel : Real default := 100.0;
    }

    enum def Color {
        enum orange; enum yellow; enum green;
        enum blue;   enum indigo; enum violet;
        enum silver; enum gold;   enum red;
    }
}
