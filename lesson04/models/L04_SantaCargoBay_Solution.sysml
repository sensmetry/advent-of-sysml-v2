package L04_SantaCargoBay_Solution {
    part def Sleigh {
        part cargoBay {
            item payload : Bag;
        }
    }

    part def Bag;

    // Solution
    part def BottomlessGiftBag specializes Bag {
        item gifts [*];
    }

    part def SantaSleigh specializes Sleigh {
        part redefines cargoBay {
            item giftBag redefines payload : BottomlessGiftBag {
                item toyTrain subsets gifts;
                item teddyBear subsets gifts;
                item candyCane subsets gifts;
            }
        }
    }
} // package L04_SantaCargoBay_Solution
