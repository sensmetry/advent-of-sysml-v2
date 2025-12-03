package AdventOfSysMLv2_Day3 {

    // Definition of a reusable "blueprint" for SantaSleigh
    // - Conventionally starts with uppercase letter 
    part def SantaSleigh {
        // How do we *define* how to build a SantaSleigh? By *using*
        // other components built according to other blueprints. 
        
        // Usage of another "blueprint" NaughtyNiceDashboard (see below)
        // - Not an instance! A type that classifies NaughtyNiceDashboards 
        //   in the context of a SantaSleigh.
        // - Often also called a feature (of the definition it is nested into)
        // - Conventionally starts with lowercase letter 
        part dashboard : NaughtyNiceDashboard;

        // Usage of an unspecified "blueprint", specified locally
        part cockpit {
            // Nested usage to specify cockpit locally
            ref item driver;
        }
    }

    // Definition of a reusable "blueprint" for NaughtyNiceDashboard
    part def NaughtyNiceDashboard;
}










