import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

# Claude generated transformer encoder decoer
# cross referenced with https://huggingface.co/blog/encoder-decoder

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, queries, keys, values, mask=None, deterministic=True):
        """Apply multi-head attention on the input data.
        
        Args:
            queries: [batch, q_length, hidden_dim]
            keys: [batch, kv_length, hidden_dim]
            values: [batch, kv_length, hidden_dim]
            mask: [batch, num_heads, q_length, kv_length], attention mask
            deterministic: if True, dropout is not applied
            
        Returns:
            output of shape [batch, q_length, hidden_dim]
        """
        batch_size, q_len, _ = queries.shape
        _, kv_len, _ = keys.shape
        
        head_dim = self.hidden_dim // self.num_heads
        
        # Linear projections
        query_proj = nn.Dense(features=self.hidden_dim, name='query_proj')(queries)
        key_proj = nn.Dense(features=self.hidden_dim, name='key_proj')(keys)
        value_proj = nn.Dense(features=self.hidden_dim, name='value_proj')(values)
        
        # Reshape for multi-head attention
        query_heads = query_proj.reshape(batch_size, q_len, self.num_heads, head_dim)
        key_heads = key_proj.reshape(batch_size, kv_len, self.num_heads, head_dim)
        value_heads = value_proj.reshape(batch_size, kv_len, self.num_heads, head_dim)
        
        # Transpose to [batch_size, num_heads, length, head_dim]
        query_heads = jnp.transpose(query_heads, (0, 2, 1, 3))
        key_heads = jnp.transpose(key_heads, (0, 2, 1, 3))
        value_heads = jnp.transpose(value_heads, (0, 2, 1, 3))
        
        # Compute attention weights
        attention_logits = jnp.matmul(query_heads, jnp.transpose(key_heads, (0, 1, 3, 2)))
        attention_logits = attention_logits / jnp.sqrt(head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attention_logits = jnp.where(mask, attention_logits, jnp.finfo(jnp.float32).min)
        
        # Apply softmax to get attention weights
        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        
        # Apply dropout to attention weights
        if not deterministic:
            attention_weights = nn.Dropout(rate=self.dropout_rate)(
                attention_weights, deterministic=deterministic)
        
        # Apply attention weights to values
        attention_output = jnp.matmul(attention_weights, value_heads)
        
        # Transpose and reshape back
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, q_len, self.hidden_dim)
        
        # Final projection
        output = nn.Dense(features=self.hidden_dim, name='output_proj')(attention_output)
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""
    hidden_dim: int
    intermediate_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, deterministic=True):
        """Apply feed-forward network to the input.
        
        Args:
            x: input of shape [batch, length, hidden_dim]
            deterministic: if True, dropout is not applied
            
        Returns:
            output of shape [batch, length, hidden_dim]
        """
        # First dense layer with GELU activation
        x = nn.Dense(features=self.intermediate_dim)(x)
        x = nn.gelu(x)
        
        # Second dense layer
        x = nn.Dense(features=self.hidden_dim)(x)
        
        # Apply dropout
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        return x

class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    hidden_dim: int
    num_heads: int
    intermediate_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """Apply encoder layer to the input.
        
        Args:
            x: input of shape [batch, length, hidden_dim]
            mask: self-attention mask
            deterministic: if True, dropout is not applied
            
        Returns:
            output of shape [batch, length, hidden_dim]
        """
        # Self-attention with layer normalization and residual connection
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(x, x, x, mask, deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual
        
        # Feed-forward with layer normalization and residual connection
        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForward(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            dropout_rate=self.dropout_rate
        )(x, deterministic)
        x = x + residual
        
        return x

class DecoderLayer(nn.Module):
    """Transformer decoder layer."""
    hidden_dim: int
    num_heads: int
    intermediate_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, encoder_output, self_mask=None, cross_mask=None, deterministic=True):
        """Apply decoder layer to the input.
        
        Args:
            x: input of shape [batch, length, hidden_dim]
            encoder_output: output from encoder of shape [batch, src_length, hidden_dim]
            self_mask: self-attention mask
            cross_mask: cross-attention mask
            deterministic: if True, dropout is not applied
            
        Returns:
            output of shape [batch, length, hidden_dim]
        """
        # Self-attention with layer normalization and residual connection
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(x, x, x, self_mask, deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual
        
        # Cross-attention with layer normalization and residual connection
        residual = x
        x = nn.LayerNorm()(x)
        x = MultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(x, encoder_output, encoder_output, cross_mask, deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual
        
        # Feed-forward with layer normalization and residual connection
        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForward(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            dropout_rate=self.dropout_rate
        )(x, deterministic)
        x = x + residual
        
        return x

class TransformerEncoder(nn.Module):
    """Transformer encoder."""
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    max_len: int = 512
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        """Apply transformer encoder to the input.
        
        Args:
            x: input token indices of shape [batch, length]
            mask: attention mask
            deterministic: if True, dropout is not applied
            
        Returns:
            encoder output of shape [batch, length, hidden_dim]
        """
        # Token embedding
        token_embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim
        )(x)
        
        # Positional embedding
        batch_size, seq_len = x.shape
        positions = jnp.arange(seq_len)[None, :]
        position_embedding = nn.Embed(
            num_embeddings=self.max_len,
            features=self.hidden_dim
        )(positions)
        
        # Combine embeddings and apply dropout
        x = token_embedding + position_embedding
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        # Apply encoder layers
        for i in range(self.num_layers):
            x = EncoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                intermediate_dim=self.intermediate_dim,
                dropout_rate=self.dropout_rate
            )(x, mask, deterministic)
        
        # Final layer normalization
        x = nn.LayerNorm()(x)
        
        return x

class TransformerDecoder(nn.Module):
    """Transformer decoder."""
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    max_len: int = 512
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, encoder_output, self_mask=None, cross_mask=None, deterministic=True):
        """Apply transformer decoder to the input.
        
        Args:
            x: input token indices of shape [batch, length]
            encoder_output: output from encoder of shape [batch, src_length, hidden_dim]
            self_mask: self-attention mask
            cross_mask: cross-attention mask
            deterministic: if True, dropout is not applied
            
        Returns:
            decoder output of shape [batch, length, vocab_size]
        """
        # Token embedding
        token_embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim
        )(x)
        
        # Positional embedding
        batch_size, seq_len = x.shape
        positions = jnp.arange(seq_len)[None, :]
        position_embedding = nn.Embed(
            num_embeddings=self.max_len,
            features=self.hidden_dim
        )(positions)
        
        # Combine embeddings and apply dropout
        x = token_embedding + position_embedding
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        # Apply decoder layers
        for i in range(self.num_layers):
            x = DecoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                intermediate_dim=self.intermediate_dim,
                dropout_rate=self.dropout_rate
            )(x, encoder_output, self_mask, cross_mask, deterministic)
        
        # Final layer normalization
        x = nn.LayerNorm()(x)
        
        # Output projection
        x = nn.Dense(features=self.vocab_size)(x)
        
        return x

class TransformerEncoderDecoder(nn.Module):
    """Transformer encoder-decoder model."""
    src_vocab_size: int
    tgt_vocab_size: int
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_dim: int = 2048
    max_len: int = 512
    dropout_rate: float = 0.1
    
    def setup(self):
        self.encoder = TransformerEncoder(
            vocab_size=self.src_vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            intermediate_dim=self.intermediate_dim,
            max_len=self.max_len,
            dropout_rate=self.dropout_rate
        )
        
        self.decoder = TransformerDecoder(
            vocab_size=self.tgt_vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            intermediate_dim=self.intermediate_dim,
            max_len=self.max_len,
            dropout_rate=self.dropout_rate
        )
    
    def __call__(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, 
                 cross_mask=None, deterministic=True):
        """Apply transformer encoder-decoder to the input.
        
        Args:
            src_tokens: source token indices of shape [batch, src_len]
            tgt_tokens: target token indices of shape [batch, tgt_len]
            src_mask: source attention mask
            tgt_mask: target attention mask (for causal/autoregressive attention)
            cross_mask: cross-attention mask
            deterministic: if True, dropout is not applied
            
        Returns:
            output logits of shape [batch, tgt_len, tgt_vocab_size]
        """
        encoder_output = self.encoder(src_tokens, src_mask, deterministic)
        decoder_output = self.decoder(tgt_tokens, encoder_output, tgt_mask, 
                                      cross_mask, deterministic)
        
        return decoder_output
    
    def encode(self, src_tokens, src_mask=None, deterministic=True):
        """Encode source tokens."""
        return self.encoder(src_tokens, src_mask, deterministic)
    
    def decode(self, tgt_tokens, encoder_output, tgt_mask=None, cross_mask=None, 
               deterministic=True):
        """Decode target tokens given encoded source."""
        return self.decoder(tgt_tokens, encoder_output, tgt_mask, cross_mask, deterministic)

# Helper functions for creating masks
def create_padding_mask(sequences, pad_token=0):
    """Create padding mask for transformer.
    
    Args:
        sequences: input sequences of shape [batch, length]
        pad_token: padding token id
    
    Returns:
        mask of shape [batch, 1, 1, length] where padding tokens are masked
    """
    mask = (sequences != pad_token)
    # Add dimensions for multi-head attention broadcasting
    mask = mask[:, None, None, :]
    return mask

def create_causal_mask(length):
    """Create causal mask for transformer decoder.
    
    Args:
        length: sequence length
    
    Returns:
        causal mask of shape [1, 1, length, length]
    """
    # Create mask that allows each position to attend only to previous positions
    mask = jnp.triu(jnp.ones((length, length), dtype=bool), k=1)
    mask = ~mask
    mask = mask[None, None, :, :]
    return mask

# Example: Initializing the model and generating dummy forward pass
def initialize_model(src_vocab_size=10000, tgt_vocab_size=10000):
    """Initialize the transformer encoder-decoder model with random parameters."""
    model = TransformerEncoderDecoder(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        intermediate_dim=2048,
        max_len=512,
        dropout_rate=0.1
    )
    
    # Initialize parameters with dummy input
    rng = jax.random.PRNGKey(SEED + 192831)
    # rng, dropout_rng = jax.random.split(rng)
    # init_rngs = {'params': rng, 'dropout': dropout_rng}

    batch_size = 2
    src_len = 16
    tgt_len = 20
    
    src_tokens = jnp.ones((batch_size, src_len), dtype=jnp.int32)
    tgt_tokens = jnp.ones((batch_size, tgt_len), dtype=jnp.int32)
    
    # Create masks
    src_padding_mask = create_padding_mask(src_tokens)
    tgt_padding_mask = create_padding_mask(tgt_tokens)
    tgt_causal_mask = create_causal_mask(tgt_len)
    
    # Combine padding and causal mask for decoder self-attention
    tgt_mask = jnp.logical_and(tgt_padding_mask, tgt_causal_mask)
    
    # Cross attention mask
    cross_mask = src_padding_mask
    
    params = model.init(rng, src_tokens, tgt_tokens, src_padding_mask, 
                        tgt_mask, cross_mask, deterministic=True)
    
    return model, params

def compute_loss(logits, targets, padding_mask):
    """
    Compute cross entropy loss with label smoothing.
    
    Args:
        logits: Model output logits of shape [batch, seq_len, vocab_size]
        targets: Target token ids of shape [batch, seq_len]
        padding_mask: Mask of shape [batch, seq_len] where 1=valid token, 0=padding
    
    Returns:
        loss: Scalar loss value
    """
    # Convert targets to one-hot with label smoothing
    vocab_size = logits.shape[-1]
    smoothing = 0.1
    
    # Create one-hot targets
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    
    # Apply label smoothing
    smooth_targets = one_hot_targets * (1.0 - smoothing) + smoothing / vocab_size
    
    # Compute cross entropy loss
    loss = -jnp.sum(smooth_targets * jax.nn.log_softmax(logits), axis=-1)
    
    # Apply padding mask (exclude padding tokens from loss)
    loss = loss * padding_mask
    
    # Normalize by the number of non-padding tokens
    normalizer = jnp.sum(padding_mask) + 1e-8
    loss = jnp.sum(loss) / normalizer
    
    return loss

PAD_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MAX_SEQ_LENGTH = 402
WARMUP_STEPS = 4000
SEED = 169575

def create_train_state(model, learning_rate):
    """Create training state with optimizer."""
    # Exponential decay learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=WARMUP_STEPS
    )

    # TODO: use the inverse time/sqrt decay function instead 
    # because apparently thats what they want for transformers
    decay_fn = optax.exponential_decay(
        init_value=learning_rate,
        decay_rate=0.5,
        transition_steps=10000  
    )
    
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[WARMUP_STEPS]
    )
    
    # Create optimizer with weight decay
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=schedule_fn,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
            weight_decay=0.01
        )
    )
    
    # Initialize model with dummy inputs
    rng = jax.random.PRNGKey(SEED)
    # rng, dropout_rng = jax.random.split(rng)
    # init_rngs = {'params': rng, 'dropout': dropout_rng}

    dummy_src = jnp.ones((2, MAX_SEQ_LENGTH), dtype=jnp.int32)
    dummy_tgt = jnp.ones((2, MAX_SEQ_LENGTH), dtype=jnp.int32)
    
    # Create masks for dummy inputs
    dummy_src_mask = create_padding_mask(dummy_src, PAD_TOKEN_ID)
    dummy_tgt_mask = jnp.logical_and(
        create_padding_mask(dummy_tgt, PAD_TOKEN_ID),
        create_causal_mask(MAX_SEQ_LENGTH)
    )
    
    variables = model.init(
        rng, 
        dummy_src, 
        dummy_tgt, 
        dummy_src_mask, 
        dummy_tgt_mask, 
        dummy_src_mask,
        deterministic=True
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )


# Training step
@jax.jit
def train_step(state, src_batch, tgt_batch):
    """JIT-compiled training step."""
    # Prepare inputs and targets
    # Input to decoder is target sequence shifted right (teacher forcing)
    decoder_inputs = jnp.roll(tgt_batch, shift=1, axis=1)
    decoder_inputs = decoder_inputs.at[:, 0].set(SOS_TOKEN_ID)
    
    # Create masks
    src_padding_mask = create_padding_mask(src_batch, PAD_TOKEN_ID)
    tgt_padding_mask = create_padding_mask(decoder_inputs, PAD_TOKEN_ID)
    causal_mask = create_causal_mask(MAX_SEQ_LENGTH)
    
    # Combine padding and causal masks for decoder
    tgt_mask = jnp.logical_and(tgt_padding_mask, causal_mask)
    
    # For calculating loss (flat mask)
    tgt_loss_mask = (tgt_batch != PAD_TOKEN_ID).astype(jnp.float32)

    # TODO: make this actual rng
    dropout_rng = jax.random.PRNGKey(SEED + 192831)
    
    def loss_fn(params):
        """Loss function for gradients."""
        logits = state.apply_fn(
            {'params': params},
            src_batch,
            decoder_inputs,
            src_padding_mask,
            tgt_mask,
            src_padding_mask,
            deterministic=False,
            rngs={'dropout':dropout_rng}
        )
        
        loss = compute_loss(logits, tgt_batch, tgt_loss_mask)
        return loss
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Apply gradients
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss

if __name__ == "__main__":
    
    # sample training for action trajectory
    # 6 * 6 = 36 pairwise action states + 3 padding tokens

    model, params = initialize_model(src_vocab_size=39, tgt_vocab_size=39)
    print("Transformer Encoder-Decoder model initialized successfully.")

    sample_encoding = jnp.array([[SOS_TOKEN_ID] + [0 for _ in range(400)] + [EOS_TOKEN_ID]])
    
    src, tgt = sample_encoding.copy(), sample_encoding.copy()
    
    state = create_train_state(model, 0.0001)

    state, loss = train_step(state, src, tgt)

    print(loss)

    # now do multiple training steps with lax.scan

    def train_step_wrapper(state, batch):
        src, tgt = batch
        state, loss = train_step(state, src, tgt)        
        return state, loss

    sample_batch = jnp.array([src.copy(), tgt.copy()])
    inputs = jnp.array([sample_batch.copy() for _ in range(10)])

    final_state, losses = jax.lax.scan(train_step_wrapper, state, inputs, length=10)
    
    print(losses)