from denver.learners import OnenetLearner
from denver.temperature_scaling import ModelWithTeperature


def train():
    model_path ='models/denver-onenet-v0.9.tar.gz'

    onenet_learner = OnenetLearner(mode='inference', model_path=model_path)

    model = ModelWithTeperature(learner=onenet_learner)

    logits, logits_sc, labels, temperature = model.set_temperature(
        valid_data='data/lta/ttest.csv', 
        text_col='text', 
        intent_col='intent', 
        tag_col='tags', 
        shuffle=True,
        lowercase=True, 
        rm_emoji=True,
        rm_url=True,
        rm_special_token=False
    )

    model.save_model(model_dir='./models', model_name='denver-onenet-scaling-v0.9.2.tar.gz')

    model.visualize(logits, logits_sc, labels, save_dir='./visualize')

def infer():
    model_path ='models/denver-onenet-v0.9.2.tar.gz'
    onenet_learner = OnenetLearner(mode='inference', model_path=model_path)
    model = ModelWithTeperature(learner=onenet_learner)
    model.load_model(model_path='models/denver-onenet-scaling-v0.9.2.tar.gz')

    out = model.process(sample=':)', lowercase=True, rm_emoji=True, rm_url=True, rm_special_token=True)

    from pprint import pprint
    pprint(out)


# train()
infer()