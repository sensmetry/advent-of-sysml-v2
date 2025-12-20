package L20_CI_CD_Automation {
        private import NumericalFunctions::*;
        private import SequenceFunctions::*;
        private import ScalarValues::*;
        private import USCustomaryUnits::*;
        private import ISQ::*;
        private import SI::*;

    package SleighDesign {
        attribute maxSafeLoad subsets ISQ::mass = 375 [kg] {
            doc DescriptionEN locale "en-US" /* Maximum safe load per reindeer. */
            doc DescriptionFR locale "fr-FR" /* Charge maximale sécuritaire par renne. */
        }

        part def SantaSleigh {
            doc DescriptionEN locale "en-US" /* The main delivery vehicle for Christmas Eve operations. */
            doc DescriptionFR locale "fr-FR" /* Le véhicule principal de livraison pour les opérations de la veille de Noël. */

            // Attributes of the sleigh
            constant attribute sleighWeight subsets ISQ::mass := 850 [kg];

            // Components of the sleigh
            part cockpit {
                doc DescriptionEN locale "en-US" /* Santa's command center during flight. */
                doc DescriptionFR locale "fr-FR" /* Centre de commande du Père Noël pendant le vol. */
                ref item driver = santa;
            }
            part cargoBay : CargoBay {
                doc DescriptionEN locale "en-US" /* Storage area for presents during delivery runs. */
                doc DescriptionFR locale "fr-FR" /* Zone de stockage des cadeaux pendant les tournées de livraison. */
            }
            part dashboard : NaughtyNiceDashboard {
                doc DescriptionEN locale "en-US" /* Real-time tracking system for children's behavior status. */
                doc DescriptionFR locale "fr-FR" /* Système de suivi en temps réel du comportement des enfants. */
            }

            // Entities related to the sleigh
            ref item santa {
                doc DescriptionEN locale "en-US" /* The jolly pilot and chief delivery officer. */
                doc DescriptionFR locale "fr-FR" /* Le pilote joyeux et chef des livraisons. */
                attribute weight subsets ISQ::mass default 120 [kg];
            }
            abstract ref part reindeer : Reindeer [9] = (dasher, dancer, prancer, vixen, comet, cupid, donner, blitzen, rudolph);

            ref part dasher : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* The speedster of the team, always first off the line. */
                doc DescriptionFR locale "fr-FR" /* Le sprinteur de l'équipe, toujours premier au départ. */
                constant attribute :>> noseColor = Color::orange;
                constant attribute :>> weight := 180 [kg];
                constant attribute :>> power := 5940 [W];
            }
            ref part dancer : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* Graceful and agile, performs aerial maneuvers with style. */
                doc DescriptionFR locale "fr-FR" /* Gracieux et agile, effectue des manœuvres aériennes avec style. */
                constant attribute :>> noseColor = Color::yellow;
                constant attribute :>> weight := 175 [kg];
                constant attribute :>> power := 7 [hp];
            }
            ref part prancer : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* Proud and confident, leads with high-stepping precision. */
                doc DescriptionFR locale "fr-FR" /* Fier et confiant, mène avec précision. */
                constant attribute :>> noseColor = Color::green;
                constant attribute :>> weight := 185 [kg];
                constant attribute :>> power := 5550 [W];
            }
            ref part vixen : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* Clever navigator with sharp instincts for weather patterns. */
                doc DescriptionFR locale "fr-FR" /* Navigateur rusé avec un instinct aiguisé pour les conditions météorologiques. */
                constant attribute :>> noseColor = Color::blue;
                constant attribute :>> weight := 160 [kg];
                constant attribute :>> power := 4.8 [kW];
            }
            ref part comet : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* Blazing through the sky, brings wonder to children below. */
                doc DescriptionFR locale "fr-FR" /* Flamboyant à travers le ciel, émerveille les enfants en bas. */
                constant attribute :>> noseColor = Color::indigo;
                constant attribute :>> weight := 195 [kg];
                constant attribute :>> power := 6142 [W];
            }
            ref part cupid : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* Spreads joy and warmth, keeping team morale high. */
                doc DescriptionFR locale "fr-FR" /* Répand la joie et la chaleur, maintient le moral de l'équipe. */
                constant attribute :>> noseColor = Color::violet;
                constant attribute :>> weight := 190 [kg];
                constant attribute :>> power := 5.7 [kW];
            }
            ref part donner : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* Thunder incarnate, the powerhouse of the team. */
                constant attribute :>> noseColor = Color::silver;
                constant attribute :>> weight := 230 [kg];
                constant attribute :>> power := 7935 [W];
            }
            ref part blitzen : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* Lightning-fast reflexes, surges through stormy skies. */
                constant attribute :>> noseColor = Color::gold;
                constant attribute :>> weight := 225 [kg];
                constant attribute :>> power := 7762 [W];
            }
            ref part rudolph : Reindeer subsets reindeer {
                doc DescriptionEN locale "en-US" /* The legendary guide whose glowing nose illuminates the darkest nights. */
                constant attribute :>> noseColor = Color::red;
                attribute :>> energyLevel := 0.2 [kW];
                constant attribute :>> weight := 100 [kg];
                constant attribute :>> power := 4950 [W];
            }

            // Calculations
            attribute reindeerCount : Natural = size(reindeer);
            attribute totalPower subsets ISQ::power := sum(reindeer.power);
            attribute averagePower subsets ISQ::power := totalPower / reindeerCount;

            attribute totalWeight subsets ISQ::mass := sleighWeight + sum(reindeer.weight) + santa.weight + cargoBay.weight;
            attribute loadPerReindeer subsets ISQ::mass := totalWeight / reindeerCount;
            attribute safetyMargin subsets ISQ::mass := maxSafeLoad - loadPerReindeer;

            attribute flightMode : FlightMode := if totalWeight > 2500 [kg] ? FlightMode::HeavyHaul else FlightMode::AgileMode;
            attribute maxSpeed = if flightMode == FlightMode::AgileMode ? 140.4 ['km/h'] else 70.2 [km / h];
        }

        part def Reindeer {
            constant attribute noseColor : Color;
            attribute energyLevel subsets ISQ::power default 100.0 [W];
            constant attribute weight subsets ISQ::mass default 110 [kg];
            constant attribute power subsets ISQ::power default 5 [kW];
        }

        part def NaughtyNiceDashboard;

        abstract part def WeightedObject {
            constant attribute weight subsets ISQ::mass;
        }

        abstract part def CargoBay {
            attribute weight subsets ISQ::mass := sum(payload.totalWeight);
            attribute bagCount : Natural := size(payload);
            attribute giftCount : Natural := sum(payload.giftCount);
            item payload : Bag [*];
        }

        abstract item def Bag {
            attribute totalWeight subsets ISQ::mass := sum(gifts.weight);
            attribute giftCount : Natural := size(gifts);
            item gifts [*] : WeightedObject;
        }

        enum def FlightMode {
            AgileMode;
            HeavyHaul;
        }

        enum def Color {
            enum orange;
            enum yellow;
            enum green;
            enum blue;
            enum indigo;
            enum violet;
            enum silver;
            enum gold;
            enum red;
        }
    }

    package GiftDefinitions {
        private import SleighDesign::WeightedObject;

        part def ToyCar :> WeightedObject { constant attribute :>> weight := 300 [g]; }
        part def TeddyBear :> WeightedObject { constant attribute :>> weight := 450 [g]; }
        part def Doll :> WeightedObject { constant attribute :>> weight := 400 [g]; }
        part def LEGOSet :> WeightedObject { constant attribute :>> weight := 1.2 [kg]; }
        part def ActionFigure :> WeightedObject { constant attribute :>> weight := 150 [g]; }
        part def BoardGame :> WeightedObject { constant attribute :>> weight := 800 [g]; }
        part def Puzzle :> WeightedObject { constant attribute :>> weight := 500 [g]; }
        part def Bicycle :> WeightedObject { constant attribute :>> weight := 12 [kg]; }
        part def ScooterBoard :> WeightedObject { constant attribute :>> weight := 8 [kg]; }
        part def VideoGameConsole :> WeightedObject { constant attribute :>> weight := 2.5 [kg]; }
        part def Tablet :> WeightedObject { constant attribute :>> weight := 500 [g]; }
        part def Book :> WeightedObject { constant attribute :>> weight := 350 [g]; }
        part def ArtSupplyKit :> WeightedObject { constant attribute :>> weight := 1.5 [kg]; }
        part def RemoteControlCar :> WeightedObject { constant attribute :>> weight := 2 [kg]; }
        part def MusicKeyboard :> WeightedObject { constant attribute :>> weight := 3 [kg]; }
        part def ScienceKit :> WeightedObject { constant attribute :>> weight := 1.8 [kg]; }
        part def Skateboard :> WeightedObject { constant attribute :>> weight := 3.5 [kg]; }
        part def DollHouse :> WeightedObject { constant attribute :>> weight := 5 [kg]; }
        part def TrainSet :> WeightedObject { constant attribute :>> weight := 4 [kg]; }
        part def StuffedAnimal :> WeightedObject { constant attribute :>> weight := 300 [g]; }
    }

    package DeliveryPlans {
        part def SantaSleighDeliveryPlan {
            
            part deliveryRun1382 : SleighDesign::SantaSleigh {

                ref item :>> santa {
                    attribute :>> weight = 100 [kg];
                }

                part :>> cargoBay {
                    item :>> payload = (northPoleBag, europeRouteBag, americasRouteBag, asiaRouteBag, africaOceaniaBag);

                    item northPoleBag : SleighDesign::Bag {
                        doc /* Local deliveries around the North Pole region. */
                        item :>> gifts = (
                            bicycle1, legoSet1, videoGameConsole1,
                            toyCar1, teddyBear1, book1,
                            actionFigure1, puzzle1, doll1,
                            tablet1, skateboard1, stuffedAnimal1
                        );
                        ref part bicycle1 : GiftDefinitions::Bicycle;
                        ref part legoSet1 : GiftDefinitions::LEGOSet;
                        ref part videoGameConsole1 : GiftDefinitions::VideoGameConsole;
                        ref part toyCar1 : GiftDefinitions::ToyCar;
                        ref part teddyBear1 : GiftDefinitions::TeddyBear;
                        ref part book1 : GiftDefinitions::Book;
                        ref part actionFigure1 : GiftDefinitions::ActionFigure;
                        ref part puzzle1 : GiftDefinitions::Puzzle;
                        ref part doll1 : GiftDefinitions::Doll;
                        ref part tablet1 : GiftDefinitions::Tablet;
                        ref part skateboard1 : GiftDefinitions::Skateboard;
                        ref part stuffedAnimal1 : GiftDefinitions::StuffedAnimal;
                    }

                    item europeRouteBag : SleighDesign::Bag {
                        doc /* Delivery route covering European countries. */
                        item :>> gifts = (
                            dollHouse1, scienceKit1, remoteControlCar1,
                            legoSet2, book2, actionFigure2,
                            toyCar2, musicKeyboard1, artSupplyKit1,
                            doll2, puzzle2, tablet2,
                            boardGame1, boardGame2
                        );
                        ref part dollHouse1 : GiftDefinitions::DollHouse;
                        ref part scienceKit1 : GiftDefinitions::ScienceKit;
                        ref part remoteControlCar1 : GiftDefinitions::RemoteControlCar;
                        ref part legoSet2 : GiftDefinitions::LEGOSet;
                        ref part book2 : GiftDefinitions::Book;
                        ref part actionFigure2 : GiftDefinitions::ActionFigure;
                        ref part toyCar2 : GiftDefinitions::ToyCar;
                        ref part musicKeyboard1 : GiftDefinitions::MusicKeyboard;
                        ref part artSupplyKit1 : GiftDefinitions::ArtSupplyKit;
                        ref part doll2 : GiftDefinitions::Doll;
                        ref part puzzle2 : GiftDefinitions::Puzzle;
                        ref part tablet2 : GiftDefinitions::Tablet;
                        ref part boardGame1 : GiftDefinitions::BoardGame;
                        ref part boardGame2 : GiftDefinitions::BoardGame;
                    }

                    item americasRouteBag : SleighDesign::Bag {
                        doc /* Delivery route covering North and South America. */
                        item :>> gifts = (
                            trainSet1, scooterBoard1, videoGameConsole2,
                            legoSet3,
                            actionFigure3, skateboard2, scienceKit2,
                            stuffedAnimal2, doll3, book4,
                            puzzle3
                        );
                        ref part trainSet1 : GiftDefinitions::TrainSet;
                        ref part scooterBoard1 : GiftDefinitions::ScooterBoard;
                        ref part videoGameConsole2 : GiftDefinitions::VideoGameConsole;
                        ref part legoSet3 : GiftDefinitions::LEGOSet;
                        ref part actionFigure3 : GiftDefinitions::ActionFigure;
                        ref part skateboard2 : GiftDefinitions::Skateboard;
                        ref part scienceKit2 : GiftDefinitions::ScienceKit;
                        ref part stuffedAnimal2 : GiftDefinitions::StuffedAnimal;
                        ref part doll3 : GiftDefinitions::Doll;
                        ref part book4 : GiftDefinitions::Book;
                        ref part puzzle3 : GiftDefinitions::Puzzle;
                    }

                    item asiaRouteBag : SleighDesign::Bag {
                        doc /* Delivery route covering Asian countries. */
                        item :>> gifts = (
                            bicycle2, musicKeyboard2, remoteControlCar2,
                            legoSet4, tablet3, artSupplyKit2,
                            toyCar3, book3, book5, actionFigure4,
                            scienceKit3, boardGame3, puzzle4,
                            teddyBear2, teddyBear3
                        );
                        ref part bicycle2 : GiftDefinitions::Bicycle;
                        ref part musicKeyboard2 : GiftDefinitions::MusicKeyboard;
                        ref part remoteControlCar2 : GiftDefinitions::RemoteControlCar;
                        ref part legoSet4 : GiftDefinitions::LEGOSet;
                        ref part tablet3 : GiftDefinitions::Tablet;
                        ref part artSupplyKit2 : GiftDefinitions::ArtSupplyKit;
                        ref part toyCar3 : GiftDefinitions::ToyCar;
                        ref part book3 : GiftDefinitions::Book;
                        ref part book5 : GiftDefinitions::Book;
                        ref part actionFigure4 : GiftDefinitions::ActionFigure;
                        ref part scienceKit3 : GiftDefinitions::ScienceKit;
                        ref part boardGame3 : GiftDefinitions::BoardGame;
                        ref part puzzle4 : GiftDefinitions::Puzzle;
                        ref part teddyBear2 : GiftDefinitions::TeddyBear;
                        ref part teddyBear3 : GiftDefinitions::TeddyBear;
                    }

                    item africaOceaniaBag : SleighDesign::Bag {
                        doc /* Delivery route covering Africa and Oceania. */
                        item :>> gifts = (
                            dollHouse2, trainSet2, scooterBoard2,
                            videoGameConsole3, bicycle3, musicKeyboard3,
                            legoSet5, book6, book7,
                            actionFigure5, teddyBear4
                        );
                        ref part dollHouse2 : GiftDefinitions::DollHouse;
                        ref part trainSet2 : GiftDefinitions::TrainSet;
                        ref part scooterBoard2 : GiftDefinitions::ScooterBoard;
                        ref part videoGameConsole3 : GiftDefinitions::VideoGameConsole;
                        ref part bicycle3 : GiftDefinitions::Bicycle;
                        ref part musicKeyboard3 : GiftDefinitions::MusicKeyboard;
                        ref part legoSet5 : GiftDefinitions::LEGOSet;
                        ref part book6 : GiftDefinitions::Book;
                        ref part book7 : GiftDefinitions::Book;
                        ref part actionFigure5 : GiftDefinitions::ActionFigure;
                        ref part teddyBear4 : GiftDefinitions::TeddyBear;
                    }
                }
            }
        }
    }
}
