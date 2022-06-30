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
                        votes = dot(x, 1.94491750637  , -0.559465554765  , 0.910004329315  , 1.14900496438  , -1.604644789323  , -1.751360532299  , 0.790603559654  , 0.561764762219  , -1.37244215968  , 1.505546325285  , 2.392666958164  , -1.521595801379  , 0.314850753614  , 0.934056112639  , -0.722486425095  , -0.185476789896  , 1.092725052897  , -0.888584010773  , -0.857458591911  , 0.341096232392  , -0.877517217796  , 0.630537192749  , 0.818692354143  , -0.151278593365  , -0.095153449492  , -0.500433796233  , 0.268845724236  , 0.742730106449  , 1.731880529472  , 2.921995105331  , 2.016810361666  , 0.007980066721  , 0.494101726961  , 0.84960213378  , -2.425901904788 );
                        // votes += -1.023126395384;
                        votes +=  -7.682015562854;
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
                        va_start(w, 69);
                        float dot = 0.0;

                        for (uint16_t i = 0; i < 69; i++) {
                            const float wi = va_arg(w, double);
                            dot += x[i] * wi;
                        }

                        return dot;
                    }
                };
            }
        }
    }