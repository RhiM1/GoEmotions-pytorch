from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel#, BertForSequenceClassification, BertConfig, AutoModel
import os
# from typing import Callable, Optional
# import os
# from attrdict import AttrDict
# import json


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
            dropout = 0.0,
            batch_norm = False
            ):
        super(ffnn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Neural network f
        if batch_norm:
            self.ffnn_stack = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(p = dropout),
                activation,
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.Dropout(p = dropout),
                activation,
                nn.Linear(embed_dim, output_dim),
                nn.BatchNorm1d(embed_dim)
            )
        else:
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
        p_factor = 1,
        use_sm = False
    ):
        super().__init__()

        self.p_factor = p_factor
        self.use_sm = use_sm
        if use_sm:
            self.sm = nn.Softmax(dim = 1)


    def forward(self, features, ex_features, ex_class_reps, p_factor = None):
        
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
        if self.use_sm:
            a = self.sm(p_factor * s)
        else:
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
            args = None, 
            load_dir = None
        ):
        super(base_model, self).__init__()

        if load_dir is not None:
            self.load_pretrained(load_dir = load_dir)
        elif args is None:
            print("Must provide either save location or config file for loading model")
        else:
            self.args = args

    def save_pretrained(self, output_dir, epoch = None):    
        ep = f"{epoch}_" if epoch is not None else ""
        torch.save(self.args, f"{output_dir}/{ep}config.json")
        torch.save(self.state_dict(), f"{output_dir}/{ep}model.mod")
             

    def load_pretrained(self, load_dir, epoch = None):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ep = f"{epoch}_" if epoch is not None else ""
        self.args = torch.load(f"{load_dir}/{ep}config.json")
        state_dict = torch.load(f"{load_dir}/{ep}model.mod")
        self.__init__(self.args)
        self.load_state_dict(state_dict)
        print(state_dict)
        quit()



class ffnn_wrapper(base_model):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
            self,
            args = None,
            load_dir = None
            ):
        super().__init__(args, load_dir)

        # input_dim = config['input_dim']
        # embed_dim = config['embed_dim']
        # num_labels = config['num_labels']
        # dropout = config['dropout']
        self.loss_fct = nn.BCEWithLogitsLoss()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.ffnn_stack = ffnn(
            input_dim = args.input_dim,
            embed_dim = args.class_dim,
            output_dim = args.num_labels,
            dropout = args.do_class,
            batch_norm = args.use_batch_norm
        )


    def forward(self, features, labels):

        logits = self.ffnn_stack(features)
        if labels is None:
            loss = None
        else:
            loss = self.loss_fct(logits, labels)
        
        return loss, logits



class minerva(base_model):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        args,
        ex_classes = None,
        ex_features = None,
        ex_IDX = None,
        load_dir = None
    ):
        super().__init__(args = args, load_dir = load_dir)

        self.loss_fct = nn.BCEWithLogitsLoss()

        if args.use_g:
            feat_dim = args.feat_dim if args.feat_dim is not None else args.input_dim
            self.g = nn.Linear(
                in_features = self.args.input_dim,
                out_features = feat_dim,
                bias = False
            )

        if args.do_feat is not None:
            self.do_feat = nn.Dropout(p = args.do_feat)
        else:
            self.do_feat = None

        self.minerva = minerva2(p_factor = args.p_factor, use_sm = args.use_sm)

        if args.do_class is not None:
            self.do_class = nn.Dropout(p = args.do_class)
        else:
            self.do_class = None

        self.set_exemplars(ex_features, ex_classes, ex_IDX)
        self.initialise_exemplars()
        if load_dir is not None:
            self.load_pretrained(load_dir)

    def set_exemplars(self, ex_features, ex_classes, ex_IDX):
        
        self.ex_IDX = ex_IDX
        if ex_classes is None:
            self.ex_classes = None
        else:
            self.ex_classes = nn.Parameter(ex_classes, requires_grad = False)
        # self.ex_features = ex_features
        if ex_features is None:
            self.ex_features = None
        else:
            self.ex_features = nn.Parameter(ex_features, requires_grad = False)
            # print(f"ex_features init: \n{self.ex_features}")


    def initialise_exemplars(self):

        print()
        if self.args.class_dim is None:
            self.class_reps = torch.nn.Parameter(
                nn.functional.one_hot(torch.arange(self.args.num_labels)).type(torch.float),
                requires_grad = self.args.train_class_reps
            )
            if self.args.train_ex_class:
                self.add_ex_class_reps = torch.nn.Parameter(
                    torch.zeros(self.args.num_ex, self.args.num_labels, dtype = torch.float)
                )
                print("ex_classes.size:", self.ex_classes.size())
        else:
            self.class_reps = torch.nn.Parameter(
                torch.rand(self.args.num_labels, self.args.class_dim, dtype = torch.float) * 2 - 1,
                requires_grad = self.args.train_class_reps
            )
            if self.args.train_ex_class:
                self.add_ex_class_reps = torch.nn.Parameter(
                    torch.zeros(self.args.num_ex, self.args.class_dim, dtype = torch.float)
                )
                print("ex_classes.size:", self.ex_classes.size())
        
        if self.args.train_ex_feats:
            self.add_ex_feats = nn.Parameter(
                torch.zeros(len(self.ex_classes), self.args.input_dim, dtype = torch.float)
            )
        
        if self.ex_features is not None:
            print("ex_features.size:", self.ex_features.size())
        print("class_reps.size:", self.class_reps.size())
        print()


    def forward(self, features, labels = None, ex_features = None, ex_classes = None, p_factor = None):
        
        # features has dim (batch_size, input_dim)
        # ex_features has dim (ex_batch_size, input_dim)
        # ex_phone_reps has dim (ex_batch_size, phone_dim)
        # ex_reps has dim (num_classes, phone_dim)
        # print(f"features.size: {features.size()}, ex_features.size: {self.ex_features.size()}")

        ex_features = ex_features if ex_features is not None else self.ex_features
        ex_classes = ex_classes if ex_classes is not None else self.ex_classes

        if self.args.train_ex_feats:
            ex_features += self.add_ex_feats

        if self.args.use_g:
            features = self.g(features)
            ex_features = self.g(ex_features)
        # print(f"features.size: {features.size()}, ex_features.size: {ex_features.size()}")

        if self.do_feat is not None:
            features = self.do_feat(features)
            ex_features = self.do_feat(ex_features)

        ex_class_reps = torch.matmul(
            torch.nn.functional.normalize(ex_classes, dim = -1),
            self.class_reps
        )
        if self.args.train_ex_class:
            ex_class_reps += self.add_ex_class_reps
            
        if self.do_class is not None:
            ex_class_reps = self.do_class(ex_class_reps)

        # print(f"class rep dim: {class_reps.size()}")    
        # print(f"ex_classes dim: {self.ex_classes.size()}")   
        # print(f"features dim: {features.size()}")
        # print(f"ex_features dim: {ex_features.size()}")
        # print(f"ex_class_reps dim: {ex_class_reps.size()}")


        echo, _ = self.minerva(features, ex_features, ex_class_reps, p_factor)

        # probs has dim (batch_size, num_phones)
        logits = -torch.cdist(echo, self.class_reps)
        # print(f"\nneg_dists:\n{neg_dists}")

        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None
            # print(f"\nloss: {loss}")

        return loss, logits

    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.args.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))
    
    def save_pretrained(self, output_dir, epoch):    

        super().save_pretrained(output_dir, epoch)
        
        if self.ex_IDX is not None:
            torch.save(self.ex_IDX, output_dir + "/ex_IDX.pt")


    def load_pretrained(self, load_dir, epoch = None):

        super().load_pretrained(load_dir, epoch)

        if os.path.exists(load_dir + "/ex_IDX.pt"):
            self.ex_IDX = torch.load(load_dir + "/ex_IDX.pt")
        else:
            self.ex_IDX = None
    


class minerva_detEx(base_model):
    # Exemplar model incorporating multi-head attention for exemplar weighting, 
    # with separate attention for the acoustic and phonetic information.
    def __init__(
        self, 
        args,
        ex_classes = None,
        ex_features = None,
        ex_IDX = None,
        load_dir = None
    ):

        if load_dir is not None:
            self.load_pretrained(load_dir)
        else:
            super().__init__(args = args, load_dir = load_dir)

            self.loss_fct = nn.BCEWithLogitsLoss()

            if args.use_g:
                feat_dim = args.feat_dim if args.feat_dim is not None else args.input_dim
                if args.use_ffnn:
                    self.g = ffnn(
                        input_dim = self.args.input_dim,
                        embed_dim = feat_dim,
                        output_dim = feat_dim,
                        dropout = args.do_feat,
                        batch_norm = args.use_batch_norm
                    )
                else:
                    if args.use_batch_norm:
                        self.g = nn.Sequential(
                            nn.Linear(
                                in_features = self.args.input_dim,
                                out_features = feat_dim,
                                bias = False
                            ),
                            nn.BatchNorm1d(feat_dim),
                            nn.Dropout(p = args.do_feat)
                        )
                    else:
                        self.g = nn.Sequential(
                            nn.Linear(
                                in_features = self.args.input_dim,
                                out_features = feat_dim
                            ),
                            nn.Dropout(p = args.do_feat)
                        )

            self.minerva = minerva2(p_factor = args.p_factor, use_sm = args.use_sm)

            if args.do_class is not None:
                self.do_class = nn.Dropout(p = args.do_class)
            else:
                self.do_class = None

            self.set_exemplars(ex_features, ex_classes, ex_IDX)
            self.initialise_exemplars()

    def set_exemplars(self, ex_features, ex_classes, ex_IDX):
        
        self.ex_IDX = ex_IDX
        if ex_classes is None:
            self.ex_classes = None
        else:
            self.ex_classes = nn.Parameter(ex_classes, requires_grad = False)
        # self.ex_features = ex_features
        if ex_features is None:
            self.ex_features = None
        else:
            self.ex_features = nn.Parameter(ex_features, requires_grad = False)
            # print(f"ex_features init: \n{self.ex_features}")


    def initialise_exemplars(self):

        if self.args.class_dim is None:
            class_reps = torch.arange(self.args.num_labels)
            class_reps = nn.functional.one_hot(class_reps).type(torch.float)
        else:
            class_reps = torch.rand(self.args.num_labels, self.args.class_dim, dtype = torch.float)
        self.class_reps = torch.nn.Parameter(class_reps, requires_grad = self.args.train_class_reps)

        if self.args.train_ex_class:
            # ex_class_reps = class_reps[self.ex_classes]
            ex_class_reps = torch.matmul(
                torch.nn.functional.normalize(self.ex_classes, dim = -1, p = 1),
                # self.ex_classes,
                self.class_reps
            )
            self.ex_class_reps = torch.nn.Parameter(ex_class_reps)
            # print("ex_class_reps.size:", self.ex_class_reps.size())

    def forward(self, features, labels = None, ex_features = None, ex_classes = None, p_factor = None):
        
        # features has dim (batch_size, input_dim)
        # ex_features has dim (ex_batch_size, input_dim)
        # ex_phone_reps has dim (ex_batch_size, phone_dim)
        # ex_reps has dim (num_classes, phone_dim)
        # print(f"features.size: {features.size()}, ex_features.size: {self.ex_features.size()}")

        ex_features = ex_features if ex_features is not None else self.ex_features

        # if self.args.train_ex_feats:
        #     ex_features += self.add_ex_feats

        if self.args.use_g:
            features = self.g(features)
            ex_features = self.g(ex_features)
        # print(f"features.size: {features.size()}, ex_features.size: {ex_features.size()}")

        # if self.do_feat is not None:
        #     features = self.do_feat(features)
        #     ex_features = self.do_feat(ex_features)

        if ex_classes is None and self.args.train_ex_class:
            ex_class_reps = self.ex_class_reps
        elif ex_classes is None:
            ex_class_reps = torch.matmul(
                # torch.nn.functional.normalize(self.ex_classes, dim = -1),
                self.ex_classes,
                self.class_reps
            )
        else:
            ex_class_reps = torch.matmul(
                # torch.nn.functional.normalize(ex_classes, dim = -1),
                ex_classes,
                self.class_reps
            )
            
        if self.do_class is not None:
            ex_class_reps = self.do_class(ex_class_reps)

        # print(f"class rep dim: {class_reps.size()}")    
        # print(f"ex_classes dim: {self.ex_classes.size()}")   
        # print(f"features dim: {features.size()}")
        # print(f"ex_features dim: {ex_features.size()}")
        # print(f"ex_class_reps dim: {ex_class_reps.size()}")


        echo, _ = self.minerva(features, ex_features, ex_class_reps, p_factor)

        # probs has dim (batch_size, num_phones)
        logits = -torch.cdist(echo, self.class_reps)
        # print(f"\nneg_dists:\n{neg_dists}")

        if labels is not None:
            loss = self.loss_fct(logits, labels)
        else:
            loss = None
            # print(f"\nloss: {loss}")

        return loss, logits

    def activation(self, s, p_factor = None):

        if p_factor is None:
            p_factor = self.args.p_factor

        return torch.mul(torch.pow(torch.abs(s), p_factor), torch.sign(s))
    
    def save_pretrained(self, output_dir, epoch = None):    

        super().save_pretrained(output_dir, epoch)
        
        if self.ex_IDX is not None:
            torch.save(self.ex_IDX, output_dir + "/ex_IDX.pt")


    def load_pretrained(self, load_dir, epoch = None):

        ep = f"{epoch}_" if epoch is not None else ""
        args = torch.load(f"{load_dir}/{ep}config.json")
        state_dict = torch.load(f"{load_dir}/{ep}model.mod")

        ex_classes = state_dict['ex_classes'].to('cpu') if 'ex_classes' in state_dict else None
        # ex_class_reps = state_dict['ex_class_reps'] if 'ex_class_reps' in state_dict else None
        ex_features = state_dict['ex_features'].to('cpu') if 'ex_features' in state_dict else None

        if os.path.exists(load_dir + "/ex_IDX.pt"):
            ex_IDX = torch.load(load_dir + "/ex_IDX.pt")
        else:
            ex_IDX = None

        self.__init__(
            args,
            ex_classes = ex_classes,
            ex_features = ex_features,
            ex_IDX = ex_IDX
        )

        self.load_state_dict(state_dict)
        self.to(self.args.device)
    



class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel(config)

        self.classifier = ffnn(
            input_dim = config.hidden_size,
            embed_dim = args.class_dim,
            output_dim = config.num_labels,
            dropout = args.do_class
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

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            # outputs = (loss,) + outputs

        return loss, logits
        # return outputs  # (loss), logits, (hidden_states), (attentions)



class BertMinervaForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self, config, args, exemplars = None, ex_IDX = None):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.exemplars = exemplars

        self.bert = BertModel(config)
        self.minerva = minerva(
            args,
            ex_classes = exemplars[3] if exemplars is not None else None,
            ex_IDX = ex_IDX
        )
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
            exemplars = None,
            labels=None,
    ):

        exemplars = exemplars if exemplars is not None else self.exemplars

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
            input_ids = exemplars[0],
            attention_mask = exemplars[1],
            token_type_ids = exemplars[2]
        )
        pooled_exemplar_output = exemplar_output[1]

        loss, logits = self.minerva(
            features = pooled_output, 
            ex_features = pooled_exemplar_output,
            ex_classes = exemplars[3] if self.minerva.ex_classes is None else None,
            labels = labels
        )

        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # if labels is not None:
        #     loss = self.loss_fct(echos, labels)
        # outputs = (loss,) + outputs

        return loss, logits
        # return outputs  # (loss), logits, (hidden_states), (attentions)
    
    def save_pretrained(self, save_directory: str, is_main_process: bool = True, state_dict: dict or None = None, save_function = torch.save, push_to_hub: bool = False, max_shard_size: int or str = "10GB", safe_serialization: bool = False, **kwargs):
        self.minerva.save_pretrained(save_directory)
        return super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, **kwargs)


