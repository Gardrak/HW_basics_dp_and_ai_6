from settings import * 


# 1. Токенизатор
def create_tokenizer(text, save_path="tokenizer.json"):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    
    tokenizer.train_from_iterator([text], trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer



# 2. Датасет
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokens = tokenizer.encode(text).ids
        self.blocks = []
        
        for i in range(0, len(tokens) - block_size + 1, block_size):
            block = tokens[i:i+block_size]
            if len(block) == block_size:
                self.blocks.append(block)
    
    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        return torch.tensor(self.blocks[idx], dtype=torch.long)



# 3. Модель Transformer Decoder
class GeneratorTransformer(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.token_to_id("[EOS]")
        self.bos_token_id = tokenizer.token_to_id("[BOS]")
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_embedding = nn.Embedding(MAX_LENGTH, D_MODEL)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        
        self.fc = nn.Linear(D_MODEL, VOCAB_SIZE)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        # Создаем маску для авторегрессии
        mask = self.generate_square_mask(seq_len).to(x.device)
        
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.decoder(
            tgt=x,
            memory=x,
            tgt_mask=mask
        )
        return self.fc(self.dropout(x))
    
    def generate_square_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    
    def generate(self, prompt, temperature=1.0, max_out_tokens=200):
        self.eval()
        with torch.no_grad():
            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt).ids],
                device=DEVICE
            )
            
            generated = input_ids.clone()
            eos_detected = False
            
            for _ in range(max_out_tokens):
                # Обрезаем контекст до max_length
                input_trunc = generated[:, -MAX_LENGTH:]
                
                logits = self(input_trunc)[0, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.eos_token_id:
                    eos_detected = True
                    break
            
            return self.tokenizer.decode(generated[0].tolist())


# 4. Обучение модели
def train_model():
    # Загрузка данных
    source = SOURCE
    text = requests.get(source).text[500:10000]  # Используем часть текста для демонстрации
    
    # Инициализация токенизатора
    tokenizer = create_tokenizer(text)
    
    # Подготовка датасета
    dataset = TextDataset(text, tokenizer, MAX_LENGTH )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Инициализация модели
    model = GeneratorTransformer(tokenizer).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_token_id)
    
    # Цикл обучения
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            inputs = batch.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs[:, :-1])
            loss = criterion(
                outputs.view(-1, VOCAB_SIZE),
                inputs[:, 1:].contiguous().view(-1)
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model


# 5. Интерфейс для тестирования
def chat():
    model = train_model()
    model.eval()
    
    print("Бот готов к общению! Введите 'quit' для выхода")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'quit':
            break
        response = model.generate(user_input, temperature=0.8)
        print(f"Бот: {response}")




if __name__ == "__main__":
    chat()