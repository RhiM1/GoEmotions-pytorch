import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification, BertConfig


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        print(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



class minerva(nn.Module):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        ex_features,
        ex_classes,
        ex_IDX,
        dim_reduce = None,
        class_dim = None,
        use_g = True,
        dropout = 0,
        p_factor = 1,
        train_class_rep = False,
        train_ex_class = False,
        train_ex_feat = False,
        num_classes = 39
    ):
        super().__init__()

        self.dim_reduce = dim_reduce
        self.class_dim = class_dim
        self.use_g = use_g
        self.dropout = dropout
        self.p_factor = p_factor
        self.train_class_reps = train_class_rep
        self.train_ex_class = train_ex_class
        self.train_ex_feat = train_ex_feat
        self.num_classes = num_classes
        
        if dim_reduce == None:
            dim_reduce = ex_features.size()[1]

        if use_g:
            self.g = nn.Linear(
                in_features = ex_features.size()[1],
                out_features = dim_reduce,
                bias = False
            )

        if dropout > 0:
            self.do = nn.Dropout(p = dropout)
        self.set_exemplars(ex_features, ex_classes, ex_IDX)
        self.initialise_exemplars()

    def set_exemplars(self, ex_features, ex_classes, ex_IDX):
        
        self.ex_classes = nn.Parameter(ex_classes.type(torch.float), requires_grad = False)
        self.ex_features = nn.Parameter(ex_features, requires_grad = False)
        self.ex_IDX = nn.Parameter(ex_IDX, requires_grad = False)


    def initialise_exemplars(self):

        if self.class_dim is None:
            class_reps = nn.functional.one_hot(torch.arange(self.num_classes)).type(torch.float)
            if self.train_ex_class:
                add_ex_reps = torch.zeros(len(self.ex_classes), self.num_classes, dtype = torch.float)
        else:
            class_reps = torch.rand(self.num_classes, self.num_classes, dtype = torch.float)
            if self.train_ex_class:
                add_ex_reps = torch.zeros(len(self.ex_classes), self.num_classes, dtype = torch.float)

        self.base_class_reps = nn.Parameter(class_reps, requires_grad = False)

        if self.train_class_reps:
            self.add_class_reps = nn.Parameter(torch.zeros_like(self.base_class_reps))

        if self.train_ex_class:
            self.add_ex_reps = nn.Parameter(add_ex_reps)

        if self.train_ex_feat:
            add_feats = torch.zeros(len(self.ex_features), self.dim_reduce, dtype = torch.float)
            self.add_feats = nn.Parameter(add_feats)
        
        print()
        print("ex_features.size:", self.ex_features.size())
        print("ex_phones.size:", self.ex_classes.size())
        print("class_reps.size:", self.base_class_reps.size())
        print()



    def forward(self, features, p_factor = None):
        
        # features has dim (batch_size, input_dim)
        # ex_features has dim (ex_batch_size, input_dim)
        # ex_phone_reps has dim (ex_batch_size, phone_dim)
        # ex_reps has dim (num_classes, phone_dim)

        if p_factor is None:
            p_factor = self.p_factor

        if self.use_g:
            features = self.g(features)
            ex_features = self.g(self.ex_features)
        else:
            ex_features = self.ex_features

        if self.train_ex_feat:
            ex_features += self.add_feats
        
        if self.train_class_reps:
            class_reps = self.base_class_reps + self.add_class_reps
        else:
            class_reps = self.base_class_reps

        if self.train_ex_class:
            # ex_class_reps = class_reps[self.ex_classes] + self.add_ex_reps
            ex_class_reps = self.ex_classes + self.add_ex_reps
        else:
            # ex_class_reps = class_reps[self.ex_classes]
            ex_class_reps = self.ex_classes

        if self.dropout > 0:
            features = self.do(features)
            ex_features = self.do(ex_features)

        # print(f"class rep dim: {class_reps.size()}")    
        # print(f"ex_classes dim: {self.ex_classes.size()}")   
        # print(f"features dim: {features.size()}")
        # print(f"ex_features dim: {ex_features.size()}")
        # print(f"ex_class_reps dim: {ex_class_reps.size()}")


        # s has dim (batch_size, ex_batch_size)
        s = torch.matmul(
            nn.functional.normalize(features, dim = 1), 
            torch.t(nn.functional.normalize(ex_features, dim = 1))
        )
        # print(f"s dim: {s.size()}")

        # a has dim (batch_size, ex_batch_size)
        # a = torch.pow(s, self.p_factor)
        a = self.activation(s, p_factor)
        # print(f"a dim: {a.size()}")

        # intensity = torch.sum(a, dim = 1)

        # a = nn.functional.normalize(a, dim = 1, p = 1)

        # echo has dim (batch_size, phone_dim)
        echo = torch.matmul(a, ex_class_reps)

        b = ex_class_reps.sum(dim = 0)
        # b = a.sum(dim = 1)
        # print(f"b size: {b.size()}")

        echo = torch.div(echo, b)

        # probs has dim (batch_size, num_phones)
        probs = -torch.cdist(echo, class_reps)

        return probs, echo, nn.functional.normalize(s, dim = 1, p = 1)

    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))





class minerva_mse(nn.Module):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        config,
        ex_classes
    ):
        super().__init__()

        self.input_dim = config.input_dim
        self.num_classes = config.num_classes
        self.use_g = config.use_g
        self.dim_reduce = config.dim_reduce
        self.dropout = config.dropout
        self.p_factor = config.p_factor
        self.train_class_reps = config.train_class_reps
        self.train_ex_class = config.train_ex_class
        
        if self.dim_reduce == None:
            self.dim_reduce = self.input_dim

        if self.use_g:
            self.g = nn.Linear(
                in_features = self.input_dim,
                out_features = self.dim_reduce,
                bias = False
            )

        if self.dropout > 0:
            self.do = nn.Dropout(p = self.dropout)

        self.class_reps = nn.Parameter(nn.functional.one_hot(torch.arange(self.num_classes)), requires_grad = self.train_class_reps)

        self.ex_classes = nn.Parameter(ex_classes, requires_grad = self.train_ex_class)


    def forward(self, features, ex_features, p_factor = None):
        
        if p_factor is None:
            p_factor = self.p_factor

        if self.use_g:
            features = self.g(features)
            ex_features = self.g(ex_features)

        if self.dropout > 0:
            features = self.do(features)
            ex_features = self.do(ex_features)

        # Cosine similarity
        # s has dim (batch_size, ex_batch_size)
        s = torch.matmul(
            nn.functional.normalize(features, dim = 1), 
            torch.t(nn.functional.normalize(ex_features, dim = 1))
        )

        # Activation
        # a has dim (batch_size, ex_batch_size)
        a = self.activation(s, p_factor)
        # print(f"a dim: {a.size()}")

        # Intensity
        # intensity has dim batch_size
        # intensity = torch.sum(a, dim = 1)

        # echo has dim (batch_size, num_classes)
        echo = torch.matmul(a, self.ex_classes)


        b = self.ex_classes.sum(dim = 0)
        b = torch.clamp(b, min = 1)
        # # b = a.sum(dim = 1)
        # # print(f"b size: {b.size()}")

        echo = torch.div(echo, b)
        echo = torch.clamp(echo, min = 0, max = 1)
        # print(f"echo:\n\t{echo}")

        # # probs has dim (batch_size, num_phones)
        # probs = -torch.cdist(echo, class_reps)

        return echo

    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))






class minerva_base(nn.Module):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        config,
        ex_classes
    ):
        super().__init__()

        self.input_dim = config.input_dim
        self.num_classes = config.num_classes
        self.use_g = config.use_g
        self.dim_reduce = config.dim_reduce
        self.dropout = config.dropout
        self.p_factor = config.p_factor
        self.train_ex_class = config.train_ex_class
        
        if self.dim_reduce == None:
            self.dim_reduce = self.input_dim

        if self.use_g:
            self.g = nn.Linear(
                in_features = self.input_dim,
                out_features = self.dim_reduce,
                bias = False
            )

        if self.dropout > 0:
            self.do = nn.Dropout(p = self.dropout)

        self.ex_classes = nn.Parameter(ex_classes, requires_grad = self.train_ex_class)


    def forward(self, features, ex_features, p_factor = None):
        
        if p_factor is None:
            p_factor = self.p_factor

        if self.use_g:
            features = self.g(features)
            ex_features = self.g(ex_features)

        if self.dropout > 0:
            features = self.do(features)
            ex_features = self.do(ex_features)

        # Cosine similarity
        # s has dim (batch_size, ex_batch_size)
        s = torch.matmul(
            nn.functional.normalize(features, dim = 1), 
            torch.t(nn.functional.normalize(ex_features, dim = 1))
        )

        # Activation
        # a has dim (batch_size, ex_batch_size)
        a = self.activation(s, p_factor)
        # print(f"a dim: {a.size()}")

        # Intensity
        # intensity has dim batch_size
        # intensity = torch.sum(a, dim = 1)

        # echo has dim (batch_size, num_classes)
        echo = torch.matmul(a, self.ex_classes)


        b = self.ex_classes.sum(dim = 0)
        b = torch.clamp(b, min = 1)
        # # b = a.sum(dim = 1)
        # # print(f"b size: {b.size()}")

        echo = torch.div(echo, b)
        echo = torch.clamp(echo, min = 0, max = 1)
        # print(f"echo:\n\t{echo}")

        # # probs has dim (batch_size, num_phones)
        # probs = -torch.cdist(echo, class_reps)

        return echo

    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))



class MinervaConfig:
    r"""
    Documentation here
    """

    def __init__(
        self,
        input_dim = None,
        dim_reduce = None,
        class_dim = None,
        use_g = True,
        dropout = 0,
        p_factor = 1,
        train_class_reps = False,
        train_ex_class = False,
        train_ex_feat = False,
        num_classes = 39
    ):
        # super().__init__(self)
        self.input_dim = input_dim
        self.dim_reduce = dim_reduce
        self.class_dim = class_dim
        self.use_g = use_g
        self.dropout = dropout
        self.p_factor = p_factor
        self.train_class_reps = train_class_reps
        self.train_ex_class = train_ex_class
        self.train_ex_feat = train_ex_feat
        self.num_classes = num_classes


class BertMinervaForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, minerva_config, exemplars):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.exemplars = exemplars
        self.minerva = minerva_base(minerva_config, ex_classes = exemplars[3])
        self.loss_fct = nn.BCELoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        exemplar_output = self.bert(
            input_ids = self.exemplars[0],
            attention_mask = self.exemplars[1],
            token_type_ids = self.exemplars[2]
        )
        pooled_exemplar_output = exemplar_output[1]

        pooled_output = self.dropout(pooled_output)
        echos = self.minerva(pooled_output, pooled_exemplar_output)

        outputs = (echos,) + outputs[2:]  # add hidden states and attention if they are here


        if labels is not None:
            loss = self.loss_fct(echos, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    

    
class BertMinervaMSEForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, minerva_config, exemplars):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.exemplars = exemplars
        self.minerva = minerva_base(minerva_config, ex_classes = exemplars[3])
        self.class_reps = nn.Parameter(
            nn.functional.one_hot(torch.arange(config.num_labels)).type(torch.float), 
            requires_grad = minerva_config.train_class_reps
        )
        self.loss_fct = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        exemplar_output = self.bert(
            input_ids = self.exemplars[0],
            attention_mask = self.exemplars[1],
            token_type_ids = self.exemplars[2]
        )
        pooled_exemplar_output = exemplar_output[1]

        pooled_output = self.dropout(pooled_output)
        echos = self.minerva(pooled_output, pooled_exemplar_output)
        distances = torch.cdist(echos, self.class_reps)

        outputs = (distances, echos,) + outputs[2:]  # add hidden states and attention if they are here


        if labels is not None:
            print(f"\nechos size:\n{echos.size()}\n")
            print(f"\nechos size:\n{echos.size()}\n")
            loss = self.loss_fct(echos, self.class_reps)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)