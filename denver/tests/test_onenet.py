from pprint import pprint
from denver.data import DenverDataSource
from denver.learners import OnenetLearner
from denver.trainers.trainer import ModelTrainer

def test_train():

    train_path = './data/lu/train.csv'
    test_path = './data/lu/test.csv'

    data_source = DenverDataSource.from_csv(train_path=train_path, test_path=test_path, 
                                        text_col='text', intent_col='intent', tag_col='tag')

    onenet = OnenetLearner(mode='training', data_source=data_source,
                           word_pretrained_embedding='vi-glove-50d')

    # onenet.train(base_path='./models/onenet/', num_epochs=20)

    trainer = ModelTrainer(learn=onenet)
    trainer.train(base_path='./models/onenet/', learning_rate=0.001, batch_size=64, num_epochs=20)

    # o = onenet.process(sample="Xe đẩy naddle con k sh", lowercase=True)
    # pprint(o)

def test_inference():
    model_path = './models/onenet/model.tar.gz'
    onenet = OnenetLearner(mode='inference', model_path=model_path)

    o = onenet.process(
        sample="xe đẩy naddle 2019 trọng lượng bn vậy shop", lowercase=True)
    pprint(o)

def test_validate():
    train_path = './data/lu/train.csv'
    test_path = './data/lu/test.csv'

    data_source = DenverDataSource.from_csv(train_path=train_path, test_path=test_path,
                                            text_col='text', intent_col='intent', tag_col='tag')

    onenet = OnenetLearner(mode='training', data_source=data_source,
                           word_pretrained_embedding='vi-glove-50d')

    onenet.train(base_path='./models/onenet/', num_epochs=1)

    o = onenet.validate()
    pprint(o)

def test_evaluate():

    test_path = './data/lu/test.csv'

    model_path = './models/onenet/model.tar.gz'
    onenet = OnenetLearner(mode='inference', model_path=model_path)

    metrics = onenet.evaluate(data=test_path, text_col='text', 
                            intent_col='intent', tag_col='tag', lowercase=True)

    pprint(metrics)

def test_prediction_on_df(data='./data/cometv3/test.csv'):
    model_path = './models/onenet2/denver-onenet.tar.gz'
    onenet = OnenetLearner(mode='inference', model_path=model_path)

    data_df = onenet.predict_on_df(data=data, intent_col='intent', tag_col='tag', lowercase=True)

    data_df.to_csv("./data/cometv3/pred-test.csv", encoding='utf-8', index=False)

def test_get_spans():

    wordss = [['B', 'ơi', 'set', 'khăn', 'yodo', 'có', 'sẵn', 'màu', 'gì'],
             ['khăn', 'yodo'], 
             ['B', 'ơi', 'set', 'khăn', 'yodo'], 
             ['B', 'ơi', 'set', 'khăn', 'yodo', '123'], 
             ['khăn', 'tam', 'yodo', 'có', 'sẵn', 'màu', 'gì']]

    tagss = [['O', 'O', 'O', 'B-inform#object_type', 'B-inform#brand', 'O', 'O', 'O', 'O'], 
            ['B-inform#object_type', 'B-inform#brand'], 
            ['O', 'O', 'O', 'B-inform#object_type', 'B-inform#brand'], 
            ['O', 'O', 'O', 'B-inform#object_type', 'B-inform#brand', 'I-inform#brand'], 
            ['B-inform#object_type', 'O', 'B-inform#brand', 'O', 'O', 'O', 'O']]

    for i in range(len(wordss)):
        words = wordss[i]
        tags = tagss[i]

        start = 0
        entity = None
        end = 0
        ping = False
        spans = []

        for i in range(len(tags)):
            if ping == True:
                if tags[i] == 'O':
                    end = i
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))
                    ping = False

                elif ("B-" in tags[i]) and (i == len(tags) - 1):
                    # append the current span tags
                    end = i
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))
                    start = i

                    # append the lastest span tags
                    entity = tags[i][2:]
                    end = i + 1
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))

                elif "B-" in tags[i]:
                    end = i
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))
                    ping = True
                    start = i
                    entity = tags[i][2:]

                elif i == len(tags) - 1:
                    end = i+1
                    value = ' '.join(words[start:end]).strip()
                    spans.append((entity, str(f"{entity} ({start}, {end-1}): {value}")))

            else:
                if "B-" in tags[i]:
                    start = i
                    entity = tags[i][2:]
                    ping = True
            
        print(spans)
    
# test_train()
# test_inference()
# test_validate()
# test_evaluate()
# test_prediction_on_df()

# test_get_spans()