# DehCode

**FREE-FIRST runtime/router local para modelos abertos de vídeo por IA.**

O DehCode está sendo construído para responder: “qual é a maneira mais simples e barata de executar este modelo no meu hardware?”. Ele reúne diagnóstico, catálogo declarativo, recomendações conservadoras, download e uma camada de execução com adapters.

## Estado real: alpha experimental

O pacote instala e a CLI funciona. Diagnóstico, registro, validação dos parâmetros e contratos de cache foram testados em Linux sem GPU. O adapter Wan usa uma implementação real de Diffusers, mas **ainda não houve download completo dos pesos nem geração validada em GPU neste projeto**. Não é um MVP comprovado ponta a ponta.

Não há benchmarks próprios, promessa de compatibilidade, usuários inventados, quantização implementada ou interface simulada. UI, segundo modelo e ComfyUI estão planejados. [Progresso e próximo passo](docs/progress.md).

## Por que existe

Usar modelos de vídeo localmente costuma exigir combinar CUDA, PyTorch, checkpoints e scripts diferentes. A proposta é concentrar esse trabalho em adapters e runtimes extensíveis. Nenhuma API paga, assinatura, conta cloud ou ComfyUI é requisito do caminho básico. Hardware, energia, armazenamento e conexão ainda têm custos.

## Instalar e diagnosticar

Python 3.10 ou superior. Python 3.10/3.12 fazem parte da matriz de CI configurada; a validação local foi em 3.12.

```bash
git clone https://github.com/Uniscz/DehCode.git
cd DehCode
python -m venv .venv
```

Ative o ambiente:

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -e .
dehcode doctor
dehcode models
dehcode recommend
```

Os três comandos também aceitam `--json`. Se o executável não estiver no PATH, use `python -m dehcode.cli` no lugar de `dehcode`.

## Instalar o modelo e gerar

Este primeiro adapter requer GPU NVIDIA com BF16 e PyTorch com CUDA funcional. CPU-only, AMD e Apple Silicon não são caminhos de geração implementados. O diagnóstico continua utilizável nesses sistemas.

Instale uma distribuição PyTorch compatível com seu sistema/driver pelo [seletor oficial](https://pytorch.org/get-started/locally/). DehCode não altera drivers nem instala CUDA de sistema automaticamente.

```bash
dehcode install wan-1.3b --dependencies
dehcode doctor
dehcode run wan-1.3b --prompt "A cat walking through a sunlit garden" --output outputs/cat.mp4
```

`--dependencies` instala o extra de vídeo declarado no pacote dentro do Python ativo. Sem essa opção, o instalador espera as dependências existentes. Também é possível executar `python -m pip install -e ".[video]"` primeiro. Os pesos são grandes; o instalador consulta o tamanho e verifica espaço de maneira conservadora.

Parâmetros disponíveis: `--negative-prompt`, `--width`, `--height`, `--frames`, `--fps`, `--seed`, `--steps`, `--guidance`, `--gpu`, `--profile` e `--output`.

```bash
# Vertical 9:16; configuração experimental, sem garantia de qualidade
dehcode run wan-1.3b --prompt "A person walking in the rain, cinematic lighting" --width 432 --height 768 --frames 81 --fps 16 --seed 42 --profile model-offload --output outputs/vertical.mp4
```

O progresso vai para stderr. O resultado informa o caminho do MP4. Um JSON ao lado registra parâmetros e revisão dos pesos, incluindo o prompt. Não publique esse arquivo se o prompt for privado. Arquivos existentes não são sobrescritos.

## Modelo e configurações

| Modelo | Runtime | Precisão | Estado |
| --- | --- | --- | --- |
| Wan 2.1 T2V 1.3B | Diffusers | BF16, VAE FP32 | Adapter implementado; GPU não validada |

Fonte: [model card oficial](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers).

Os perfis são `gpu`, `model-offload` (padrão) e `sequential-offload`. Offload transfere parte da pressão para RAM e pode ser lento. Os requisitos medidos de VRAM/RAM para esta implementação são **desconhecidos**. O motor não transforma números de outra implementação em garantias. FP8 e outras quantizações não estão implementados.

O adapter valida dimensões múltiplas de 16 e contagens de frames no formato 4n+1. O limite inicial é 399.360 pixels por frame e 81 frames. Defaults: 832×480, 81 frames, 16 FPS, 50 passos, guidance 5. Alterar FPS muda a reprodução, não é um condicionamento do modelo.

## Cache e portabilidade

`DEHCODE_HOME` configura metadados e outputs; padrão `~/.cache/dehcode`. O cache respeita `HF_HUB_CACHE`, depois `HF_HOME`, ou usa o diretório DehCode. A revisão upstream é resolvida para SHA e registrada. Repetir a instalação reaproveita um snapshot completo. `--revision SHA` seleciona uma revisão explicitamente.

Pesos ficam fora do Git e não são copiados para cada geração. O Hugging Face gerencia blobs e locks de download. O estado de instalação verifica presença e tamanho dos arquivos; isso não equivale a auditoria criptográfica. A geração usa somente arquivos locais e não baixa pesos silenciosamente. Para downloads privados, use a autenticação padrão Hugging Face, nunca credenciais no código. `.env.example` é apenas documentação; DehCode não carrega `.env` automaticamente.

## Arquitetura

```text
src/dehcode/
  cli.py          comandos públicos
  core/           registry, recomendações, requests, downloads, orquestração
  hardware/       diagnóstico independente das dependências ML
  registry/       catálogo JSON com fontes e desconhecidos explícitos
  adapters/       validação e tradução específicas do modelo
  runtimes/       execução compartilhada Diffusers
  config/         diretórios configuráveis
```

O núcleo usa o protocolo de adapter e seleciona runtimes por nome. Não importa pipelines de modelo. Um novo adapter precisa ser registrado em `adapters/__init__.py` e ter entrada JSON; plugins externos automáticos ainda não existem. Veja [arquitetura](docs/architecture.md).

## Verificação e roadmap

```bash
python -m unittest discover -s tests -v
python -m pip wheel --no-deps --wheel-dir dist .
```

A CI faz testes CPU em Linux, Windows e macOS. Não inclui geração, download de pesos ou provisionamento pago. Testes com doubles conferem contratos, não qualidade do vídeo nem consumo de memória.

1. Validar instalação completa e vídeo real no Wan: [procedimento GPU](docs/gpu-validation.md).
2. Registrar versões, memória e falhas; melhorar as recomendações com evidências.
3. Adicionar segundo adapter para provar modularidade.
4. Implementar interface local com progresso e gestão de jobs.
5. Integrar ComfyUI como runtime opcional e expandir runtimes nativos.

Limitações atuais: sem GPU verificada, instalação automática de drivers, quantização, execução distribuída, cancelamento de jobs persistentes, GUI ou ComfyUI. As dependências possuem faixas de versão; um lock reproduzível de inferência será estabelecido após validação GPU.

## Licença e contribuição

O código usa [MIT](LICENSE), escolhida pela simplicidade de reutilização, modificação e distribuição, inclusive comercial. A licença do código não relicencia os pesos: o Wan indicado declara Apache-2.0. Consulte sempre a licença do modelo e dos componentes.

Veja [CONTRIBUTING](CONTRIBUTING.md), [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md), [SECURITY](SECURITY.md) e [CHANGELOG](CHANGELOG.md).
