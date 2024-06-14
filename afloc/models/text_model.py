import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np


class BertEncoder(nn.Module):
    """Text encoder of AFLoc."""
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()

        self.bert_type = cfg.model.text.bert_type
        self.last_n_layers = cfg.model.text.last_n_layers
        self.aggregate_method = cfg.model.text.aggregate_method
        self.norm = cfg.model.text.norm
        self.embedding_dim = cfg.model.text.embedding_dim
        self.freeze_bert = cfg.model.text.freeze_bert
        self.agg_tokens = cfg.model.text.agg_tokens
        self.take_sent_as_units = cfg.model.text.take_sent_as_units

        self.model = AutoModel.from_pretrained(
            self.bert_type, output_hidden_states=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.emb_global, self.emb_local = None, None

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):
        """
        Aggregate token-level embeddings to word-level embeddings

        Inputs:
            embeddings (torch.Tensor): token-level embeddings
            caption_ids (torch.Tensor): token ids

        Returns:
            agg_embs_batch (torch.Tensor): word-level embeddings
            sentences (list): text report
        """
        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):
            
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        if self.aggregate_method == "sum":
                            new_emb = new_emb.sum(axis=0)
                        elif self.aggregate_method == "mean":
                            new_emb = new_emb.mean(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, ids, attn_mask, token_type):
        """
        Forward pass of the text model

        Inputs:
            ids (torch.Tensor): input ids
            attn_mask (torch.Tensor): attention mask
            token_type (torch.Tensor): token type

        Returns:
            res (dict): 
            - word_embeddings (torch.Tensor): word-level embeddings
            - report_embeddings (torch.Tensor): report-level embeddings
            - sent_embeddings (torch.Tensor): sentence-level embeddings
            - report (list): text report
            - sent_units_report (list): an auxiliary sentence list
        """

        outputs = self.model(ids, attn_mask, token_type)

        # aggregate intermetidate layer's embeddings
        if self.last_n_layers > 1:
            all_embeddings = outputs[2] 
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  

            embeddings = embeddings.permute(1, 0, 2, 3) 

            if self.agg_tokens:
                embeddings, report = self.aggregate_tokens(embeddings, ids)
            else:
                report = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            report_embeddings = embeddings.mean(axis=2)  
            sent_embeddings = None
            sent_units_report = None
            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)  
                report_embeddings = report_embeddings.sum(axis=1)  
                if self.take_sent_as_units:
                    sent_embeddings, sent_units_report = self.aggregate_tokens_sent_as_units(embeddings, report)
                    sent_embeddings = sent_embeddings.sum(axis=1) 
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                report_embeddings = report_embeddings.mean(axis=1)
                if self.take_sent_as_units:
                    sent_embeddings, sent_units_report = self.aggregate_tokens_sent_as_units(embeddings, report)
                    sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer's embeddings
        else:
            word_embeddings, report_embeddings = outputs[0], outputs[1]

        # project word embeddings from (batch_dim, num_words, feat_dim) to (batch_dim, embedding_dim, num_words)
        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        if self.emb_local is not None:
            word_embeddings = self.emb_local(word_embeddings)
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        # project sent embeddings from (batch_dim, num_words, feat_dim) to (batch_dim, embedding_dim, num_words)
        if self.take_sent_as_units:
            batch_dim, num_words, feat_dim = sent_embeddings.shape
            sent_embeddings = sent_embeddings.view(batch_dim * num_words, feat_dim)
            if self.emb_local is not None:
                sent_embeddings = self.emb_local(sent_embeddings)
            sent_embeddings = sent_embeddings.view(batch_dim, num_words, self.embedding_dim)
            sent_embeddings = sent_embeddings.permute(0, 2, 1)

        # project report embeddings
        if self.emb_global is not None:
            report_embeddings = self.emb_global(report_embeddings)

        # normalize embeddings
        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            report_embeddings = report_embeddings / torch.norm(
                report_embeddings, 2, dim=1, keepdim=True
            ).expand_as(report_embeddings)
            if self.take_sent_as_units:
                sent_embeddings = sent_embeddings / torch.norm(
                    sent_embeddings, 2, dim=1, keepdim=True
                ).expand_as(sent_embeddings)

        res = {
            "word_embeddings": word_embeddings,
            "report_embeddings": report_embeddings,
            "sent_embeddings": sent_embeddings,
            "report": report,
            "sent_units_report": sent_units_report,
        }
        return res

    def aggregate_tokens_sent_as_units(self, embeddings, report):
        """
        Aggregate token-level embeddings to sentence-level embeddings

        Inputs:
            embeddings (torch.Tensor): token-level embeddings
            report (list): text report
            
        Returns:
            agg_embs_batch (torch.Tensor): sentence-level embeddings
            sentences (list): an auxiliary sentence list
        """
        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        
        # loop over batch
        for embs, sts in zip(embeddings, report):
            # embs: [97, 4, 768] sts: [97]
            agg_embs = []
            words = []
            cap = np.array(sts)
            idx = np.where(cap == "[SEP]") 
            cap[idx[0]-1] = "*"
            cap[idx] = "." 
            idx = np.where(cap == ".") 

            agg_embs.append(embs[0])
            words.append(f"[CLS]")
            for i in range(len(idx[0])):
                if i == 0:
                    start = 1
                    end = idx[0][i]
                else:
                    start = idx[0][i-1] + 1
                    end = idx[0][i]
                agg_embs.append(embs[start:end].mean(axis=0))
                words.append(f"sent{i}")
            words.append(f"[SEP]")
            agg_embs.append(embs[idx[0][-1]])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences