# MaiaSelfDrivingAI

Experimenting with several CNN to classify what action to take next in the racing game F1 2015.
The neural networks do not get any extra input except images.

Current Training data size: 100,000 frames

| Neural Network | Epochs till Max | Accuracy | Behaviour |
| -------------- | --------------- | -------- | --------- |
| Squeezenet     | 37              | Ca. 95%  | Likes to cut corners. Sometimes doesnt want to go forward |
| InceptionV3    | 10 (5 Head 5 Rest) | Ca. 90% | Does not drive well at all. Probably needs more training data |

Overall Comments:
Collecting more training data will most probably allow for more accurate classification

## Future Goals

- Collect more training data
- Implement vjoy control
- compare more neural networks
- Reinforcement learning
