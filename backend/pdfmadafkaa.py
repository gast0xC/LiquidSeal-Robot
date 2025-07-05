from fpdf import FPDF
from fpdf.enums import XPos, YPos  # Importação necessária para as novas posições


class EnhancedDocumentationPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.current_chapter = ""

    def header(self):
        if self.page_no() == 1:
            self.set_font("Helvetica", "B", 18)
            self.set_text_color(0, 51, 102)
            self.cell(0, 10, "Documentação - Add Variant GUI", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.ln(10)
            self.set_draw_color(0, 51, 102)
            self.set_line_width(1)
            self.line(10, 30, 200, 30)
        else:
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(50, 50, 50)
            self.cell(0, 10, self.current_chapter, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            self.set_draw_color(0, 51, 102)
            self.set_line_width(0.5)
            self.line(10, 20, 200, 20)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Página {self.page_no()}", new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")

    def add_cover(self, title, subtitle):
        self.add_page()
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(0, 51, 102)
        self.cell(0, 40, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "", 14)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, subtitle, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(30)
        self.set_font("Helvetica", "", 12)
        self.cell(0, 10, "Criado por: Hanon Systems", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.cell(0, 10, "Data: 12/2024", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(10)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(20)

    def chapter_title(self, title):
        self.add_page()
        self.current_chapter = title
        self.set_font("Helvetica", "B", 16)
        self.set_fill_color(0, 102, 204)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C", fill=True)
        self.ln(10)

    def chapter_content(self, content):
        self.set_font("Helvetica", "", 12)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 8, content.strip())
        self.ln(10)


# Conteúdo detalhado com explicações completas
sections = [
    {
        "title": "Introdução",
        "content": """
        A aplicação **Lord Waddles** foi desenvolvida para facilitar a interação com bases de dados MySQL,
        oferecendo uma interface gráfica (GUI) intuitiva. Esta ferramenta é especialmente útil para:
        - Empresas que necessitam de gerir grandes volumes de dados em bases de dados relacionais.
        - Profissionais sem conhecimentos avançados de SQL que precisam de visualizar, editar ou adicionar registos.

        **Objetivo principal**:
        Automatizar tarefas de gestão de dados e simplificar processos administrativos ligados à base de dados.

        **Vantagens da Aplicação**:
        1. Redução da dependência de conhecimentos técnicos avançados.
        2. Facilidade em visualizar e editar dados sem recorrer a comandos SQL complexos.
        3. Capacidade de trabalhar diretamente com tabelas específicas e variantes de cliente.

        **Exemplo de uso**:
        Imagine um cenário onde um cliente deseja visualizar informações específicas de um produto. A interface permite selecionar a tabela do banco de dados correspondente ao produto e aplicar filtros de variantes para extrair informações de forma ágil e eficiente.
        """
    },
    {
        "title": "Visão Geral da Aplicação",
        "content": """
        A aplicação combina várias tecnologias para oferecer uma solução robusta e eficiente.

        **Componentes Principais**:
        1. **Interface Gráfica (GUI)**:
           - Desenvolvida com a biblioteca **Tkinter**, permite interações dinâmicas com os dados.
           - Inclui menus dropdown, formulários e botões intuitivos.
        
        2. **Ligação à Base de Dados**:
           - Utiliza a biblioteca **mysql.connector** para conectar-se à base de dados MySQL.
           - Estabelece uma ligação estável e segura com o servidor.
        
        3. **Estrutura Dinâmica**:
           - A aplicação adapta-se automaticamente às tabelas selecionadas, criando formulários personalizados com base nos campos existentes.

        **Funcionalidades Oferecidas**:
        - Visualizar registos de tabelas específicas.
        - Filtrar tabelas com base em variantes de cliente.
        - Adicionar novos registos diretamente a partir da interface.
        - Atualizar registos existentes de forma eficiente.

        **Imagem da Interface**:
        A interface gráfica é projetada para ser limpa e intuitiva, com menus de fácil navegação, o que ajuda os usuários a focarem-se no que importa.
        """
    },
    {
        "title": "Estrutura do Código e Funções Detalhadas",
        "content": """
        O código está dividido em módulos para facilitar a leitura, manutenção e expansão.

        **Módulo de Conexão com a Base de Dados**:
        - Estabelece uma ligação com a base de dados MySQL utilizando as seguintes configurações:
            - **Host**: 192.168.10.42
            - **Porta**: 3377
            - **Utilizador**: jcontramestre
            - **Palavra-passe**: 904OBol6PY0mcIpN
            - **Base de Dados**: CompressorDB

        **Módulo de Extração de Dados**:
        - O módulo de extração de dados permite buscar informações de diversas tabelas. Ele é baseado em consultas SQL dinâmicas que utilizam o comando `UNION ALL`, que combina dados de várias fontes sem duplicação.
        - Exemplo de consulta gerada:
        ```sql
        SELECT DISTINCT Customer_Variant FROM (
            SELECT Customer_Variant FROM CompressorDB.Customer
            UNION ALL
            SELECT Customer_Variant FROM CompressorDB.Compressor_End_Item_Checklist
        ) AS all_variants;
        ```

        **Módulo de Manipulação de Dados**:
        - A função `save_data` permite inserir ou atualizar registros na base de dados.
        - Exemplo de operação:
        ```sql
        INSERT INTO CompressorDB.Customer (Nome, Tipo)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE Nome = VALUES(Nome), Tipo = VALUES(Tipo);
        ```

        **Dicas e Boas Práticas**:
        - Use a interface gráfica para interagir com dados sem precisar de comandos SQL avançados.
        - Sempre valide os dados antes de inseri-los no banco para garantir a integridade dos registros.
        """
    },
    {
        "title": "Fluxo de Trabalho do Utilizador",
        "content": """
        **Passo 1**: Abertura da Aplicação:
        - Execute o script Python para abrir a interface gráfica.
        - Certifique-se de que o servidor MySQL está ativo e acessível.
        
        **Passo 2**: Seleção de uma Variante de Cliente:
        - Escolha uma variante a partir do menu dropdown na interface.
        - As tabelas são automaticamente filtradas para exibir apenas as que contêm registos para essa variante.
        
        **Passo 3**: Seleção de uma Tabela:
        - Clique no dropdown de tabelas para visualizar as tabelas disponíveis.
        - Escolha a tabela desejada.
        
        **Passo 4**: Visualização e Edição de Registos:
        - Os dados da tabela serão exibidos no formulário.
        - Faça as alterações desejadas nos campos e clique em 'Guardar'.

        **Passo 5**: Adicionar um Novo Registo:
        - Clique em 'Criar' para adicionar um novo registo.
        - Preencha todos os campos obrigatórios no formulário.
        - Clique novamente em 'Criar' para salvar o novo registo na base de dados.

        **Passo 6**: Encerrar a Aplicação:
        - Clique em 'Sair' para fechar a aplicação.

        **Nota**:
        O fluxo de trabalho foi projetado para ser intuitivo e ágil, facilitando a manipulação de dados, mesmo para usuários sem conhecimento avançado em SQL.
        """
    },
    {
        "title": "Exemplos Práticos",
        "content": """
        **Exemplo 1: Atualizar Registos**
        1. Selecione a variante 'Cliente X' no menu dropdown de variantes.
        2. Escolha a tabela 'CompressorDB.Customer' no menu dropdown de tabelas.
        3. No formulário que aparece, atualize o campo `Tipo` de 'Standard' para 'Premium'.
        4. Clique no botão 'Guardar' para persistir as alterações na base de dados.

        **Exemplo 2: Adicionar um Novo Registo**
        1. Escolha a tabela 'CompressorDB.Inverter_Assembly_Torque_Config'.
        2. Preencha os campos no formulário com os seguintes valores:
           - Torque_Min: 10
           - Torque_Max: 55
           - Customer_Variant: 'Novo Cliente'
        3. Clique no botão 'Criar' para adicionar o novo registo à base de dados.
        """
    },
    {
        "title": "Resolução de Problemas",
        "content": """
        **Problema 1: Erro ao conectar-se à base de dados**
        - **Causa**: Credenciais inválidas ou o servidor MySQL está inacessível.
        - **Solução**:
          1. Verifique se o servidor MySQL está ativo.
          2. Confirme as configurações de ligação no código.

        **Problema 2: Campos em falta no formulário**
        - **Causa**: Um ou mais campos obrigatórios não foram preenchidos.
        - **Solução**:
          1. Verifique os campos destacados no formulário.
          2. Preencha todos os campos antes de clicar em 'Guardar' ou 'Criar'.

        **Problema 3: A tabela não aparece no menu dropdown**
        - **Causa**: A tabela não contém registos para a variante selecionada.
        - **Solução**:
          1. Verifique se a tabela contém dados válidos.
          2. Certifique-se de que a tabela está incluída na lista de tabelas no código.
        """
    }
]
# Gerar o PDF
pdf = EnhancedDocumentationPDF()
pdf.set_auto_page_break(auto=True, margin=15)

pdf.add_cover("Guia do Utilizador", "Add Variant GUI")
for section in sections:
    pdf.chapter_title(section["title"])
    pdf.chapter_content(section["content"])

# Salvar o PDF
output_path = "Guia_de_Utilizador_Atualizado.pdf"
try:
    pdf.output(output_path)
    print(f"PDF criado com sucesso: {output_path}")
except Exception as e:
    print(f"Erro ao criar o PDF: {e}")