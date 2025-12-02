package Lesson1_ChristmasTree {
    part def SensmetryChristmasTree {
        doc
        /* A festive Christmas tree with various decorations and components
         * - Tree parts
         * - Decorative parts
         */

        // attribute imgUrl = "file://./../images/myChristmasTree.jpg";
        
        part trunk {
            doc
            /* The main woody stem that supports the tree */

            part branches [5..10] {
                doc
                /* Primary branches extending from the trunk */

                part needles [*] {
                    doc
                    /* Pine needles covering the branches */
                }
            }
        }

        part sphericalOrnament [15] {
            attribute color : Color = Color::blue;
        }

        part starOrnament [1] {
            attribute color : Color = Color::golden;
        }

        part bellOrnament [10] {
            attribute color : Color = Color::red;
        }

        part lights [100] {
            doc
            /* String of LED lights wrapped around the tree */
        }

        part base {
            doc
            /* Stand that holds the tree upright */
        }
    }

    enum def Color {
        white;
        red;
        blue;
        green;
        golden;
    }
}
