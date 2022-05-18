#pragma once
#include <cstdarg>
#include <stdint.h>
#include <stdio.h>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class LogisticRegression {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        // float votes[2] = { -1.023126395384  };
                        float votes;
                        votes = dot(x,   0.692173132801  , -1.123761722585  , -1.137344152047  , -1.253240534497  , 0.277350277221  , 0.328165320132  , 0.204784537998  , 0.396244663798  , 0.073598438895  , 1.701410753327  , 3.072514529462  , -0.528309843003  , 0.167615667558  , -0.238384384386  , 0.341524989341  , -0.613754451299  , 0.210576854205  , -2.746509802716  , 1.935340118047  , -1.700651122505  , -0.739505730856  , 1.624020709124  , -1.080312427932  , 0.819411598876  , -3.281890866902  , 0.246589890208  , 0.560419860853  , -2.389543525524  , 1.995421021052  , 1.484409973041  , 3.258545808898  , 1.908689773952  , 0.564338543598  , -0.323757994601  , 2.832678192061  , -1.664067153985  , 0.295578171341  , -2.012978745358  , -0.785283093723  , -0.89174495494  , 0.069723670119  , -2.660592771346  , 0.191431764377  , 0.587016274209  , -0.071577729502  , -1.385374265761  , 1.443770274467  , 0.300166365762  , -4.355222153213  , -1.394917104103  , 2.054734902748  , -4.620631454629  , -2.246643368444 );
                        votes += -1.023126395384;
                        printf("%f\n", votes);

                        // return argmax of votes
                        uint8_t classIdx = 0;
                        // float maxVotes = votes[0];
                        // printf("%f\n", votes[0]);
                        // for (uint8_t i = 1; i < 2; i++) {
                        //     if (votes[i] > maxVotes) {
                        //         classIdx = i;
                        //         maxVotes = votes[i];
                        //     }
                        // }
                        if (votes > 0)
                            classIdx = 1;
                        return classIdx;
                    }

                protected:
                    /**
                    * Compute dot product
                    */
                    float dot(float *x, ...) {
                        va_list w;
                        va_start(w, 53);
                        float dot = 0.0;

                        for (uint16_t i = 0; i < 53; i++) {
                            const float wi = va_arg(w, double);
                            dot += x[i] * wi;
                        }

                        return dot;
                    }
                };
            }
        }
    }