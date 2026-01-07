import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
import numpy as np
import timm
import os
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.backends.cudnn as cudnn
from pyts.image import GramianAngularField
from sklearn.feature_extraction.text import TfidfVectorizer
import torchvision.models as models
import torchvision.transforms.functional as TF

class ImageBatchDataset(Dataset):
    def __init__(self, npy_batch_paths):
        self.npy_batch_paths = npy_batch_paths
        self._cache = {} 


        self._total_samples = 0
        self._batch_start_indices = [0]
        

        print("Mapeando arquivos .npy para índices...")
        for path in tqdm(self.npy_batch_paths):

            data = np.load(path, mmap_mode='r') 
            batch_size = data.shape[0]
            self._total_samples += batch_size
            self._batch_start_indices.append(self._total_samples)

    def __len__(self):
        return self._total_samples

    def _find_batch_and_offset(self, idx):

        batch_idx = np.searchsorted(self._batch_start_indices, idx, side='right') - 1

        offset = idx - self._batch_start_indices[batch_idx]
        return batch_idx, offset

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    TARGET_SIZE = (224, 224) 



    def __getitem__(self, idx):
        batch_idx, offset = self._find_batch_and_offset(idx)
        batch_path = self.npy_batch_paths[batch_idx]

        if batch_path not in self._cache:
            self._cache[batch_path] = np.load(batch_path)


        sample_rgb_nchw = self._cache[batch_path][offset]
        

        image_tensor = torch.tensor(sample_rgb_nchw, dtype=torch.float32)
        

        image_tensor = TF.resize(image_tensor, self.TARGET_SIZE, antialias=True)
        

        image_tensor = (image_tensor - self.IMAGENET_MEAN) / self.IMAGENET_STD

        return image_tensor

class TextInferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# --- A CLASSE DO ENSEMBLE (SIMPLIFICADA PARA 2 MODELOS) ---
class Domain_Ensemble:
    def __init__(self, cnn_path, tabular_path, deberta_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vision_model = timm.create_model('convnext_nano', pretrained=False, num_classes=2)
        print(f"Carregando checkpoint visão de: {cnn_path}")
        """self.vision_model = models.resnet18(weights=None)
        self.vision_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.vision_model.maxpool = nn.Identity()
        num_features = self.vision_model.fc.in_features
        self.vision_model.fc = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(num_features, 1)
        )
        self.vision_model.stem[0] = nn.Conv2d(3, 80, kernel_size=3, stride=1, padding=1)
        self.vision_model.stages[1].downsample[1].stride = (1, 1)
        self.vision_model.stages[2].downsample[1].stride = (1, 1)
        self.vision_model.stages[3].downsample[1].stride = (1, 1)"""


        print(f"Carregando checkpoint visão de: {cnn_path}")
        try:
            checkpoint = torch.load(cnn_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict_to_load = checkpoint['model_state_dict']
                print(optimal_threshold)
            else:
                state_dict_to_load = checkpoint

            self.vision_model.load_state_dict(state_dict_to_load)

        except Exception as e:
             raise RuntimeError(f"Erro ao carregar o checkpoint da CNN: {e}")
        self.vision_model.to(self.device)
        self.vision_model.eval()

        print("Carregando Especialista 1: DeBERTa...")
        self.deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_path, num_labels=2)
        self.deberta_model.to(self.device)
        self.deberta_model.eval()
        self.deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_path)


        print("Carregando Especialista 2: Modelo Tabular...")
        self.tabular_model = joblib.load(tabular_path) 


        self.clf = LogisticRegression(random_state=42, max_iter=1000) 
        print("Modelos base carregados. Meta-modelo: Logistic Regression.")
        
    def _get_base_predictions(self, X_image, X_tabular, X_text_raw, batch_size=256):
        """ Gera previsões dos 2 modelos base. """
        if isinstance(X_image, torch.Tensor):
            if X_image.dim() == 3: 
                X_image = X_image.unsqueeze(0)
            image_loader = DataLoader(TensorDataset(X_image), batch_size=batch_size)

        if isinstance(X_image, list):
            temp_dataset = ImageBatchDataset(X_image) 

        elif isinstance(X_image, np.ndarray):
            image_tensor = torch.tensor(X_image, dtype=torch.float32)
            temp_dataset = TensorDataset(image_tensor)
        else:

            temp_dataset = TensorDataset(X_image.float())

        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)
        
        all_features = []
        with torch.no_grad(): 
            for batch in tqdm(temp_loader, desc="Vision Predict"):
                image_batch = batch[0].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'): 
                    features_batch = self.vision_model(image_batch)
                all_features.append(torch.softmax(features_batch, dim=-1).detach().cpu().numpy())
        
        preds_vision = np.concatenate(all_features, axis=0)[:, 1].reshape(-1, 1)

        print("  Gerando previsões do DeBERTa...")
        text_dataset = TextInferenceDataset(X_text_raw, self.deberta_tokenizer)
        text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)
        all_preds_deberta = []
        with torch.no_grad():
            for batch in tqdm(text_loader, desc="  DeBERTa predict"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
                    outputs = self.deberta_model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

                all_preds_deberta.append(probs)

        preds_deberta = np.concatenate(all_preds_deberta, axis=0)[:, 1].reshape(-1, 1)


        print("  Gerando previsões do modelo tabular...")
        preds_tabular = self.tabular_model.predict_proba(X_tabular)[:, 1].reshape(-1, 1)

        return preds_vision, preds_deberta, preds_tabular

    def fit(self, X_image, X_val_tabular, X_val_text_raw, y_val):
        """ Treina o meta-modelo (self.clf) usando previsões do conjunto de validação. """
        print("Gerando previsões base no conjunto de validação...")
        p_vision, p_deberta, p_tabular = self._get_base_predictions(X_image, X_val_tabular, X_val_text_raw)
        
        X_meta_train = np.hstack((p_vision, p_deberta, p_tabular))
        
        print("Treinando o meta-modelo...")
        self.clf.fit(X_meta_train, y_val)
        print("Meta-modelo treinado com sucesso!")

    def predict_proba(self, X_image, X_tabular, X_text_raw):
        """ Gera previsões finais do ensemble. """
        if not hasattr(self.clf, 'classes_'):
             raise RuntimeError("Meta-modelo não treinado. Chame .fit() primeiro.")
             
        p_vision, p_deberta, p_tabular = self._get_base_predictions(X_image, X_tabular, X_text_raw)
        X_meta = np.hstack((p_vision, p_deberta, p_tabular))
        
        return self.clf.predict_proba(X_meta)

    def predict(self, X_image, X_tabular, X_text_raw):
        probabilities = self.predict_proba(X_image, X_tabular, X_text_raw)
        return np.argmax(probabilities, axis=1)

    @classmethod
    def load_ensemble(cls, save_dir, device='cuda'):

        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Carregando ensemble de {save_dir}...")
        

        

        tabular_path = os.path.join(save_dir, 'xgb.joblib')
        

        cnn_path = os.path.join(save_dir, 'vision_model.pth') 
        deberta_path = os.path.join(save_dir, 'deberta_model')
        
        ensemble = cls(cnn_path, tabular_path, deberta_path, device=str(device))
        
        meta_model_path = os.path.join(save_dir, 'metamodel.joblib')
        ensemble.clf = joblib.load(meta_model_path)
        
        return ensemble    
