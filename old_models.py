import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification, BertConfig, AutoModel
# from sentence_transformers import SentenceTransformer
from typing import Callable, Optional
import os
from attrdict import AttrDict
import json



class ffnn(nn.Module):
    r"""Simple feed-forward neural network with a single hidden layer.

    Args:
        input_dim: size of each query sample
        embed_dim: size of each key sample (usually the same as Q_dim)
        output_dim: size of each value sample
        dropout: size of the output for each head

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{input\_dim}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{output\_dim}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = ffnn(40, 64, 39)
        >>> input = torch.randn(128, 40)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 39])
    """

    def __init__(
            self,
            input_dim = 40, 
            embed_dim = 512,
            output_dim = 39,
            activation = nn.ReLU(),
            dropout = 0.0
            ):
        super(ffnn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Neural network f
        self.ffnn_stack = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Dropout(p = dropout),
            activation,
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p = dropout),
            activation,
            nn.Linear(embed_dim, output_dim)
        )


    def forward(self, features):
        
        return self.ffnn_stack(features)



class minerva2(nn.Module):
    r"""Direct implementation of Minerva 2 model from Hintzman (1984)
    with no trained parameters.

    Args:
        p_factor: hyperaparmeter controlling how much emphasis to place
        on similarity.

    Shape:
        - Input features: :math:`(*, H)` where :math:`*` means any number of
          dimensions including none.
        - Exemplar features: :math:`(*, D, H)` where :math:`*` means any number of
          dimensions including none.
        - Exemplar features: :math:`(*, D, C)` where :math:`*` means any number of
          dimensions including none.
        - Output: :math:`(*, C)` where all but the last dimension
          are the same shape as the input.

    Examples::

        >>> m = minerva2(3)
        >>> input = torch.randn(128, 40)
        >>> ex_features = torch.randn(64, 40)
        >>> ex_classes = torch.randn(64, 4)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 4])
    """

    def __init__(
        self, 
        ex_features = None,
        ex_class_reps = None,
        p_factor = 1
    ):
        super().__init__()

        self.ex_features = ex_features
        self.ex_class_reps = ex_class_reps
        self.p_factor = p_factor


    def forward(self, features, ex_features = None, ex_class_reps = None, p_factor = None):
        
        ex_features = ex_features if ex_features is not None else self.ex_features
        ex_class_reps = ex_class_reps if ex_class_reps is not None else self.ex_class_reps
        p_factor = p_factor if p_factor is not None else self.p_factor
        # features has dim (batch_size, input_dim)
        # ex_features has dim (ex_batch_size, input_dim)
        # ex_phone_reps has dim (ex_batch_size, class_dim)
        # ex_reps has dim (num_classes, class_dim)

        # s has dim (batch_size, ex_batch_size)
        s = torch.matmul(
            nn.functional.normalize(features, dim = 1), 
            torch.t(nn.functional.normalize(ex_features, dim = 1))
        )

        # a has dim (batch_size, ex_batch_size)
        a = self.activation(s, p_factor)

        intensity = a.sum(dim = 1)

        # echo has dim (batch_size, class_dim)
        echo = torch.matmul(a, ex_class_reps)

        return echo, intensity

    
    def activation(self, s, p_factor = None):
        # Raise to a power while preserving sign

        if p_factor is None:
            p_factor = self.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))



class base_model(nn.Module):

    def __init__(
            self, 
            config = None, 
            load_dir = None
        ):
        super(base_model, self).__init__()

        if load_dir is not None:
            self.load_pretrained(load_dir = load_dir)
        elif config is None:
            print("Must provide either save location or config file for loading model")
        else:
            self.config = config

    def save_pretrained(self, output_dir):    
        torch.save(self.config, output_dir + "/config.json")
        torch.save(self.state_dict, output_dir + "/model.mod")
             

    def load_pretrained(self, load_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = torch.load(load_dir + "/config.json")
        state_dict = torch.load(load_dir + "/model.mod")
        self.load_state_dict(state_dict)



class ffnn_wrapper(base_model):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
            self,
            config = None,
            load_dir = None
            ):
        super(ffnn, self).__init__(config, load_dir)

        # input_dim = config['input_dim']
        # embed_dim = config['embed_dim']
        # num_labels = config['num_labels']
        # dropout = config['dropout']
        self.loss_fct = nn.BCEWithLogitsLoss()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ffnn_stack = ffnn(
            input_dim = config['input_dim'],
            embed_dim = config['class_dim'],
            output_dim = config['num_labels'],
            dropout = config['dropout']
        )


    def forward(self, features, labels):

        logits = self.ffnn_stack(features)
        loss = self.loss_fct(logits, labels)
        
        return loss, logits



class minerva(base_model):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        ex_classes,
        ex_features = None,
        ex_IDX = None,
        config = None,
        load_dir = None
    ):
        super().__init__(config = config, load_dir = load_dir)

        # self.config = config
        self.loss_fct = nn.BCEWithLogitsLoss()
        
        if self.config['class_dim'] == None:
            self.config['class_dim'] = self.config['input_dim']

        print(f"config:\n{self.config}")

        if config['use_g']:
            self.g = nn.Linear(
                in_features = self.config['input_dim'],
                out_features = self.config['feat_dim'],
                bias = False
            )
            print(f"g:\n{self.g}")

        if config['dropout'] > 0:
            self.do = nn.Dropout(p = config['dropout'])
        self.set_exemplars(ex_features, ex_classes, ex_IDX)
        self.initialise_exemplars()

    def set_exemplars(self, ex_features, ex_classes, ex_IDX):
        
        self.ex_classes = nn.Parameter(ex_classes.detach().type(torch.float), requires_grad = False)
        print(f"ex_classes init: \n{self.ex_classes}")
        if ex_features is None:
            self.ex_features = None
        else:
            self.ex_features = nn.Parameter(ex_features, requires_grad = False)
            print(f"ex_features init: \n{self.ex_features}")
        if ex_IDX is None:
            self.ex_IDX = None
        else:
            self.ex_IDX = nn.Parameter(ex_IDX, requires_grad = False)
            print(f"ex_IDX init: \n{self.ex_IDX}")


    def initialise_exemplars(self):

        if self.config['class_dim'] is None:
            self.class_reps = torch.nn.Parameter(
                nn.functional.one_hot(torch.arange(self.config['num_labels'])).type(torch.float),
                requires_grad = self.config['train_class_reps']
            )
            if self.config['train_ex_class']:
                self.add_ex_class_reps = torch.nn.Parameter(
                    torch.zeros(len(self.ex_classes), self.config['num_labels'], dtype = torch.float)
                )
        else:
            self.class_reps = torch.nn.Parameter(
                torch.rand(self.config['num_labels'], self.config['class_dim'], dtype = torch.float) * 2 - 1,
                requires_grad = self.config['train_class_reps']
            )
            if self.config['train_ex_class']:
                self.add_ex_class_reps = torch.nn.Parameter(
                    torch.zeros(len(self.ex_classes), self.config['class_dim'], dtype = torch.float)
                )

        # print(f"class_reps init: \n{self.class_reps}")
        # if self.config['train_ex_class']:
        #     print(f"add_ex_class_reps init: \n{self.add_ex_class_reps}")
        if self.config['train_ex_feats']:
            self.add_feats = nn.Parameter(
                torch.zeros(len(self.ex_classes), self.config['feat_dim'], dtype = torch.float)
            )
        
        print()
        if self.ex_features is not None:
            print("ex_features.size:", self.ex_features.size())
        print("ex_classes.size:", self.ex_classes.size())
        print("class_reps.size:", self.class_reps.size())
        print()


    def forward(self, features, labels, ex_features = None, p_factor = None):
        
        # features has dim (batch_size, input_dim)
        # ex_features has dim (ex_batch_size, input_dim)
        # ex_phone_reps has dim (ex_batch_size, phone_dim)
        # ex_reps has dim (num_classes, phone_dim)
        # print(f"features.size: {features.size()}, ex_features.size: {self.ex_features.size()}")

        if ex_features is None:
            ex_features = self.ex_features

        if p_factor is None:
            p_factor = self.config['p_factor']
        # print(f"p_factor: {p_factor}")

        if self.config['use_g']:
            features = self.g(features)
            ex_features = self.g(ex_features)
        # print(f"features.size: {features.size()}, ex_features.size: {ex_features.size()}")

        if self.config['train_ex_feats']:
            ex_features += self.add_feats

        if self.config['dropout'] > 0:
            features = self.do(features)
            ex_features = self.do(ex_features)

        ex_class_reps = torch.matmul(
            torch.nn.functional.normalize(self.ex_classes, p = 1, dim = -1),
            self.class_reps
        )
        if self.config['train_ex_class']:
            ex_class_reps += self.add_ex_class_reps

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

        # a has dim (batch_size, ex_batch_size)
        # a = torch.pow(s, self.p_factor)
        a = self.activation(s, p_factor)
        # print(f"\na:\n{a}")
        # print(f"a dim: {a.size()}")

        # intensity = torch.sum(a, dim = 1)

        a = nn.functional.normalize(a, dim = 1, p = 1)

        # echo has dim (batch_size, phone_dim)
        echo = torch.matmul(a, ex_class_reps)
        # print(f"\necho1:\n{echo}")

        # if self.config['class_dim'] is None:

        #     b = torch.clamp(self.ex_class_reps.sum(dim = 0), min = 1)
        #     # print(f"\nb:\n{b}")
        #     # b = a.sum(dim = 1)
        #     # print(f"b size: {b.size()}")

        #     echo = torch.div(echo, b)
        #     # print(f"\necho2:\n{echo}")

        # probs has dim (batch_size, num_phones)
        neg_dists = -torch.cdist(echo, self.class_reps)
        # print(f"\nneg_dists:\n{neg_dists}")

        loss = self.loss_fct(neg_dists, labels)
        # print(f"\nloss: {loss}")

        return loss, neg_dists

    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.config['p_factor']

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))



class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        print(config)

        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = ffnn(
            input_dim = config.hidden_size,
            embed_dim = config.hidden_size,
            output_dim = self.num_labels,
            dropout = config.hidden_dropout_prob
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

        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)




class BertMinervaForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, minerva_config, exemplars, ex_IDX):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.exemplars = exemplars
        self.minerva = minerva(ex_classes = exemplars[3], ex_IDX = ex_IDX, config = minerva_config)
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
        loss, neg_dists = self.minerva(
            features = pooled_output, 
            ex_features = pooled_exemplar_output,
            labels = labels
        )

        outputs = (neg_dists,) + outputs[2:]  # add hidden states and attention if they are here

        # if labels is not None:
        #     loss = self.loss_fct(echos, labels)
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)




# Probably not needed below here
class contrastive_loss(nn.Module):
    r"""
    Examples::

        >>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
        >>> output = torch.full([10, 64], 1.5)  # A prediction (logit)
        >>> pos_weight = torch.ones([64])  # All weights are equal to 1
        >>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        >>> criterion(output, target)  # -log(sigmoid(1.5))
        tensor(0.20...)

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

     Examples::

        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[torch.Tensor] = None, m = 0.5) -> None:
        super(contrastive_loss, self).__init__()
        self.m = m
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        # self.weight: Optional[torch.Tensor]
        # self.pos_weight: Optional[torch.Tensor]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # Input is distances from classes
        # Target is true/false for each class

        # print(f"input:\n{input}")
        # print(f"target:\n{target}")
        # print(f"input.size: {input.size()}, target.size: {target.size()}")
        # pos_loss = torch.mul(target, input)
        # neg_loss = torch.mul((1 - target), (self.m - input))
        # for i in range(10):
        #     print(f"{i} \n {input[i]} \n {target[i]} \n {pos_loss[i]} \n {neg_loss[i]}\n")


        # neg_loss = neg_loss.clamp(min = 0)
        return (torch.mul(target, input) + torch.mul((1 - target), (self.m - input).clamp(min = 0))).mean()
        # return (pos_loss + neg_loss).mean()

        # return nn.functional.binary_cross_entropy_with_logits(input, target,
        #                                           self.weight,
        #                                           pos_weight=self.pos_weight,
        #                                           reduction=self.reduction)



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
                # bias = False
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



class minerva_mse(nn.Module):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        config,
        ex_classes,
        ex_features = None
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
        self.class_dim = config.class_dim
        self.class_init = config.class_init
        self.ex_classes = nn.Parameter(ex_classes, requires_grad = False)
        self.ex_features = nn.Parameter(ex_features, requires_grad = False)

        print(f"class_init:\n{self.class_init}")
        
        if self.dim_reduce == None:
            self.dim_reduce = self.input_dim

        if self.use_g:
            self.g = nn.Linear(
                in_features = self.input_dim,
                out_features = self.dim_reduce,
                # bias = False
            )

        if self.dropout > 0:
            self.do = nn.Dropout(p = self.dropout)

        if self.class_dim is None:
            class_reps = nn.functional.one_hot(torch.arange(self.num_classes)).type(torch.float)
            if self.train_ex_class:
                ex_class_add = torch.zeros(len(ex_classes), self.num_classes, dtype = torch.float)
        else:
            if self.class_init is None:
                class_reps = torch.rand(self.num_classes, self.class_dim) * 2 - 1
            else:
                class_reps = self.class_init
            if self.train_ex_class:
                ex_class_add = torch.zeros(len(ex_classes), self.class_dim, dtype = torch.float)

        self.class_reps = nn.Parameter(class_reps, requires_grad = self.train_class_reps)

        if self.train_ex_class:
            self.ex_class_add = nn.Parameter(ex_class_add)



        # print(self.class_reps)
        # print(self.ex_classes)
        # print(nn.functional.normalize(self.ex_classes.type(torch.float), p = 1, dim = 1) @ self.class_reps)
        # quit()

    def forward(self, features, ex_features = None, ex_classes = None, p_factor = None):
        
        if ex_features is None:
            ex_features = self.ex_features

        if ex_classes is None:
            ex_classes = self.ex_classes

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
        a = nn.functional.normalize(a, dim = 1, p = 1)
        # print(f"a dim: {a.size()}")

  
        ex_class_reps = nn.functional.normalize(ex_classes.type(torch.float), p = 1, dim = 1) @ self.class_reps
        if self.train_ex_class:
            ex_class_reps += self.ex_class_add

        # Intensity
        # intensity has dim batch_size
        # intensity = torch.sum(a, dim = 1)

        # echo has dim (batch_size, num_classes)
        echo = torch.matmul(a, ex_class_reps)
        
        distances = torch.cdist(echo, self.class_reps)

        # b = self.ex_classes.sum(dim = 0)
        # b = torch.clamp(b, min = 1)
        # # b = a.sum(dim = 1)
        # # print(f"b size: {b.size()}")

        # echo = torch.div(echo, b)
        # echo = torch.clamp(echo, min = 0, max = 1)
        # print(f"echo:\n\t{echo}")

        # # probs has dim (batch_size, num_phones)
        # probs = -torch.cdist(echo, class_reps)

        return distances

    
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
        m = 0.5,
        class_init = None,
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
        self.m = m
        self.class_init = class_init
        self.num_classes = num_classes



class BertMinervaForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, minerva_config, exemplars, ex_IDX):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.exemplars = exemplars
        self.minerva = minerva(ex_classes = exemplars[3], ex_IDX = ex_IDX, config = minerva_config)
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
        loss, neg_dists = self.minerva(
            features = pooled_output, 
            ex_features = pooled_exemplar_output,
            labels = labels
        )

        outputs = (neg_dists,) + outputs[2:]  # add hidden states and attention if they are here


        # if labels is not None:
        #     loss = self.loss_fct(echos, labels)
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    

    
class BertMinervaMSEForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, minerva_config, exemplars):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.exemplars = exemplars
        self.minerva = minerva_mse(minerva_config, ex_classes = exemplars[3])

        self.loss_fct = nn.BCEWithLogitsLoss()
        # self.loss_fct = contrastive_loss(m = minerva_config.m)

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
        distances = self.minerva(pooled_output, pooled_exemplar_output)
        outputs = (-distances, ) + outputs[2:]  # add hidden states and attention if they are here


        if labels is not None:
            # print(f"\nechos size:\n{echos.size()}\n")
            # print(f"\nechos size:\n{echos.size()}\n")
            loss = self.loss_fct(-distances, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    


    
class BertMinervaMSEForMultiLabelClassification2(BertPreTrainedModel):
    def __init__(self, config, minerva_config, exemplars):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.exemplars = exemplars
        # minerva_config2 = {'input_dim': config.hidden_size,
        #           'feat_dim': minerva_config.dim_reduce,
        #           'num_labels': minerva_config.num_classes,
        #           'dropout': minerva_config.dropout,
        #           'use_g': minerva_config.use_g,
        #           'class_dim': minerva_config.class_dim,
        #           'p_factor': minerva_config.p_factor,
        #           'train_class_reps': minerva_config.train_class_reps,
        #           'train_ex_class': minerva_config.train_ex_class,
        #           'train_ex_feats': minerva_config.train_ex_feat
        #           }
        self.minerva = minerva(ex_classes = exemplars[3], config = minerva_config)

        self.loss_fct = nn.BCEWithLogitsLoss()
        # self.loss_fct = contrastive_loss(m = minerva_config.m)

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
        # distances = self.minerva(pooled_output, pooled_exemplar_output)

        loss, neg_dists = self.minerva(
            features = pooled_output, 
            ex_features = pooled_exemplar_output,
            labels = labels
        )


        outputs = (neg_dists, ) + outputs[2:]  # add hidden states and attention if they are here


        if labels is not None:
            # print(f"\nechos size:\n{echos.size()}\n")
            # print(f"\nechos size:\n{echos.size()}\n")
            loss = self.loss_fct(neg_dists, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
    



class minerva_old(base_model):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        ex_features,
        ex_classes,
        ex_IDX,
        config = None,
        load_dir = None
    ):
        super().__init__(config = config, load_dir = load_dir)

        # self.config = config
        self.loss_fct = nn.BCEWithLogitsLoss()
        
        if self.config['feat_dim'] == None:
            self.config['feat_dim'] = self.config['input_dim']

        print(f"config:\n{self.config}")

        if config['use_g']:
            self.g = nn.Linear(
                in_features = self.config['input_dim'],
                out_features = self.config['feat_dim'],
                bias = False
            )
            print(f"g:\n{self.g}")

        if config['dropout'] > 0:
            self.do = nn.Dropout(p = config['dropout'])
        self.set_exemplars(ex_features, ex_classes, ex_IDX)
        self.initialise_exemplars(ex_classes)

    def set_exemplars(self, ex_features, ex_classes, ex_IDX):
        
        self.ex_classes = nn.Parameter(ex_classes.detach().type(torch.float), requires_grad = False)
        print(f"ex_classes init: \n{self.ex_classes}")
        self.ex_features = nn.Parameter(ex_features, requires_grad = False)
        print(f"ex_features init: \n{self.ex_features}")
        self.ex_IDX = nn.Parameter(ex_IDX, requires_grad = False)
        print(f"ex_IDX init: \n{self.ex_IDX}")


    def initialise_exemplars(self, ex_classes):

        if self.config['class_dim'] is None:
            self.class_reps = nn.Parameter(
                nn.functional.one_hot(torch.arange(self.config['num_labels'])).type(torch.float),
                requires_grad = self.config['train_class_reps']
            )
            self.ex_class_reps = nn.Parameter(
                ex_classes.type(torch.float), 
                requires_grad = self.config['train_class_reps']
            )
        else:
            self.class_reps = nn.Parameter(
                torch.rand(self.config['num_labels'], self.config['class_dim'], dtype = torch.float) * 2 - 1,
                requires_grad = self.config['train_class_reps']
            )
            self.ex_class_reps = nn.Parameter(
                torch.rand(len(ex_classes), self.config['class_dim'], dtype = torch.float) * 2 - 1, 
                requires_grad = self.config['train_class_reps']
            )

        print(f"class_reps init: \n{self.class_reps}")
        print(f"ex_reps init: \n{self.ex_class_reps}")

        if self.config['train_ex_feats']:
            add_feats = torch.zeros(len(self.ex_features), self.config['feat_dim'], dtype = torch.float)
            self.add_feats = nn.Parameter(add_feats)
        
        print()
        print("ex_features.size:", self.ex_features.size())
        print("ex_phones.size:", self.ex_classes.size())
        print("class_reps.size:", self.class_reps.size())
        print()


    def forward(self, features, labels, p_factor = None):
        
        # features has dim (batch_size, input_dim)
        # ex_features has dim (ex_batch_size, input_dim)
        # ex_phone_reps has dim (ex_batch_size, phone_dim)
        # ex_reps has dim (num_classes, phone_dim)
        # print(f"features.size: {features.size()}, ex_features.size: {self.ex_features.size()}")

        if p_factor is None:
            p_factor = self.config['p_factor']
        # print(f"p_factor: {p_factor}")

        if self.config['use_g']:
            features = self.g(features)
            ex_features = self.g(self.ex_features)
        else:
            ex_features = self.ex_features
        # print(f"features.size: {features.size()}, ex_features.size: {ex_features.size()}")

        if self.config['train_ex_feats']:
            ex_features += self.add_feats

        if self.config['dropout'] > 0:
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

        # a has dim (batch_size, ex_batch_size)
        # a = torch.pow(s, self.p_factor)
        a = self.activation(s, p_factor)
        # print(f"\na:\n{a}")
        # print(f"a dim: {a.size()}")

        # intensity = torch.sum(a, dim = 1)

        a = nn.functional.normalize(a, dim = 1, p = 1)

        # echo has dim (batch_size, phone_dim)
        echo = torch.matmul(a, self.ex_class_reps)
        # print(f"\necho1:\n{echo}")

        if self.config['class_dim'] is None:

            b = torch.clamp(self.ex_class_reps.sum(dim = 0), min = 1)
            # print(f"\nb:\n{b}")
            # b = a.sum(dim = 1)
            # print(f"b size: {b.size()}")

            echo = torch.div(echo, b)
            # print(f"\necho2:\n{echo}")

        # probs has dim (batch_size, num_phones)
        neg_dists = -torch.cdist(echo, self.class_reps)
        # print(f"\nneg_dists:\n{neg_dists}")

        loss = self.loss_fct(neg_dists, labels)
        # print(f"\nloss: {loss}")

        return loss, neg_dists

    
    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.config['p_factor']

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))