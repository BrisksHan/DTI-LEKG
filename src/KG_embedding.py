from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import numpy as np
from typing import List

def get_KG_embedding_dict(triples, save_path = 'data_temp/kg_embedding.pkl', model = 'TuckER', epochs = 1, embedding_dim = 200, device='cuda:0'):#epoch 1000 or 200
    triples = np.array(triples)
    create_inverse_triples = True
    if model == 'RGCN':
        create_inverse_triples = False
    tf = TriplesFactory.from_labeled_triples(triples, create_inverse_triples = create_inverse_triples)
    tf_few = TriplesFactory.from_labeled_triples(triples[:10], create_inverse_triples = create_inverse_triples)
    print(f'model {model}, epochs {epochs}, embedding_dim {embedding_dim}, device {device}')
    trained_model = pipeline(
        training = tf,
        testing = tf_few,
        model = model,
        model_kwargs= dict(embedding_dim= embedding_dim),
        epochs = epochs,
        device = device,
        use_tqdm = True,
    )
    trained_model = trained_model.model
    entity_representation_modules: List['pykeen.nn.Representation'] = trained_model.entity_representations
    relation_representation_modules: List['pykeen.nn.Representation'] = trained_model.relation_representations
    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]
    entity_embedding_tensor = entity_embeddings(indices=None)#.detach().numpy()
    relation_embedding_tensor = relation_embeddings(indices=None)#.detach().numpy()
    entity_embedding_np = entity_embedding_tensor.cpu().detach().numpy()
    relation_embedding_np = relation_embedding_tensor.cpu().detach().numpy()
    entity_embedding_dict = tf.entity_to_id
    relation_embedding_dict = tf.relation_to_id
    if save_path != '':
        import utils
        print('save embedding result at: ', save_path)
        utils.save_to_pickle([entity_embedding_np, entity_embedding_dict, relation_embedding_np, relation_embedding_dict], save_path)
    print(f'output dim: {len(entity_embedding_np[0])}')
    return entity_embedding_np, entity_embedding_dict, relation_embedding_np, relation_embedding_dict


