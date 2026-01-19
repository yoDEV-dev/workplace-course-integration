#!/bin/bash

# AI Agents Course - Image Migration Script
# Downloads all images from Microsoft repo and external sources

set -e

OUTPUT_DIR="./images"
mkdir -p "$OUTPUT_DIR"

echo "📥 Downloading AI Agents Course images..."
echo "==========================================="

# Lesson 00 - Configuración
echo "📁 Lesson 00 - Configuración"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/forked-repo.33f27ca1901baa6a.webp" -o "$OUTPUT_DIR/lesson-00-forked-repo.webp"
curl -sL "https://github.com/user-attachments/assets/a85e776c-2edb-4331-ae5b-6bfdfb98ee0e" -o "$OUTPUT_DIR/lesson-00-github-asset.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/profile_developer_settings.410a859fe749c755.webp" -o "$OUTPUT_DIR/lesson-00-profile-developer-settings.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/fga_new_token.1c1a234afe202ab3.webp" -o "$OUTPUT_DIR/lesson-00-fga-new-token.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/token-name-expiry-date.a095fb0de6386864.webp" -o "$OUTPUT_DIR/lesson-00-token-name-expiry-date.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/token_repository_limit.924ade5e11d9d8bb.webp" -o "$OUTPUT_DIR/lesson-00-token-repository-limit.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/add_models_permissions.c0c44ed8b40fc143.webp" -o "$OUTPUT_DIR/lesson-00-add-models-permissions.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/verify_permissions.06bd9e43987a8b21.webp" -o "$OUTPUT_DIR/lesson-00-verify-permissions.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/store_token_securely.08ee2274c6ad6caf.webp" -o "$OUTPUT_DIR/lesson-00-store-token-securely.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/github_token_field.20491ed3224b5f4a.webp" -o "$OUTPUT_DIR/lesson-00-github-token-field.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/project-endpoint.8cf04c9975bbfbf1.webp" -o "$OUTPUT_DIR/lesson-00-project-endpoint.webp"

# Lesson 01 - Introducción
echo "📁 Lesson 01 - Introducción"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-1-thumbnail.d21b2c34b32d35bb.webp" -o "$OUTPUT_DIR/lesson-01-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/what-are-ai-agents.1ec8c4d548af601a.webp" -o "$OUTPUT_DIR/lesson-01-what-are-ai-agents.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/when-to-use-ai-agents.54becb3bed74a479.webp" -o "$OUTPUT_DIR/lesson-01-when-to-use-ai-agents.webp"

# Lesson 02 - Frameworks
echo "📁 Lesson 02 - Frameworks"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-2-thumbnail.c65f44c93b8558df.webp" -o "$OUTPUT_DIR/lesson-02-thumbnail.webp"
curl -sL "https://microsoft.github.io/autogen/stable/_images/architecture-standalone.svg" -o "$OUTPUT_DIR/lesson-02-autogen-architecture-standalone.svg"
curl -sL "https://microsoft.github.io/autogen/stable/_images/architecture-distributed.svg" -o "$OUTPUT_DIR/lesson-02-autogen-architecture-distributed.svg"

# Lesson 03 - Diseño
echo "📁 Lesson 03 - Diseño"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-3-thumbnail.1092dd7a8f1074a5.webp" -o "$OUTPUT_DIR/lesson-03-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/agentic-design-principles.1cfdf8b6d3cc73c2.webp" -o "$OUTPUT_DIR/lesson-03-agentic-design-principles.webp"

# Lesson 04 - Herramientas
echo "📁 Lesson 04 - Herramientas"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-4-thumbnail.546162853cb3daff.webp" -o "$OUTPUT_DIR/lesson-04-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/functioncalling-diagram.a84006fc287f6014.webp" -o "$OUTPUT_DIR/lesson-04-functioncalling-diagram.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/agent-service-in-action.34fb465c9a84659e.webp" -o "$OUTPUT_DIR/lesson-04-agent-service-in-action.webp"

# Lesson 05 - Agentic RAG
echo "📁 Lesson 05 - Agentic RAG"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-5-thumbnail.20ba9d0c0ae64fae.webp" -o "$OUTPUT_DIR/lesson-05-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/agentic-rag-core-loop.c8f4b85c26920f71.webp" -o "$OUTPUT_DIR/lesson-05-agentic-rag-core-loop.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/tool-integration.0f569710b5c17c10.webp" -o "$OUTPUT_DIR/lesson-05-tool-integration.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/self-correction.da87f3783b7f174b.webp" -o "$OUTPUT_DIR/lesson-05-self-correction.webp"

# Lesson 06 - Agentes Confiables
echo "📁 Lesson 06 - Agentes Confiables"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-6-thumbnail.a58ab36c099038d4.webp" -o "$OUTPUT_DIR/lesson-06-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/system-message-framework.3a97368c92d11d68.webp" -o "$OUTPUT_DIR/lesson-06-system-message-framework.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/understanding-threats.89edeada8a97fc0f.webp" -o "$OUTPUT_DIR/lesson-06-understanding-threats.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/human-in-the-loop.5f0068a678f62f4f.webp" -o "$OUTPUT_DIR/lesson-06-human-in-the-loop.webp"

# Lesson 07 - Planificación
echo "📁 Lesson 07 - Planificación"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-7-thumbnail.f7163ac557bea123.webp" -o "$OUTPUT_DIR/lesson-07-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/defining-goals-tasks.d70439e19e37c47a.webp" -o "$OUTPUT_DIR/lesson-07-defining-goals-tasks.webp"

# Lesson 08 - Multi-Agente
echo "📁 Lesson 08 - Multi-Agente"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-8-thumbnail.278a3e4a59137d62.webp" -o "$OUTPUT_DIR/lesson-08-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/multi-agent-group-chat.ec10f4cde556babd.webp" -o "$OUTPUT_DIR/lesson-08-multi-agent-group-chat.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/multi-agent-hand-off.4c5fb00ba6f8750a.webp" -o "$OUTPUT_DIR/lesson-08-multi-agent-hand-off.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/multi-agent-filtering.d959cb129dc9f608.webp" -o "$OUTPUT_DIR/lesson-08-multi-agent-filtering.webp"

# Lesson 09 - Metacognición
echo "📁 Lesson 09 - Metacognición"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-9-thumbnail.38059e8af1a5b71d.webp" -o "$OUTPUT_DIR/lesson-09-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/importance-of-metacognition.b381afe9aae352f7.webp" -o "$OUTPUT_DIR/lesson-09-importance-of-metacognition.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/rag-vs-context.9eae588520c00921.webp" -o "$OUTPUT_DIR/lesson-09-rag-vs-context.webp"

# Lesson 10 - Producción
echo "📁 Lesson 10 - Producción"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-10-thumbnail.2b79a30773db093e.webp" -o "$OUTPUT_DIR/lesson-10-thumbnail.webp"
curl -sL "https://langfuse.com/images/cookbook/example-autogen-evaluation/trace-tree.png" -o "$OUTPUT_DIR/lesson-10-langfuse-trace-tree.png"
curl -sL "https://langfuse.com/images/cookbook/example-autogen-evaluation/example-dataset.png" -o "$OUTPUT_DIR/lesson-10-langfuse-example-dataset.png"
curl -sL "https://langfuse.com/images/cookbook/example-autogen-evaluation/dashboard.png" -o "$OUTPUT_DIR/lesson-10-langfuse-dashboard.png"

# Lesson 11 - Protocolos
echo "📁 Lesson 11 - Protocolos"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-11-thumbnail.b6c742949cf1ce2a.webp" -o "$OUTPUT_DIR/lesson-11-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/mcp-diagram.e4ca1cbd551444a1.webp" -o "$OUTPUT_DIR/lesson-11-mcp-diagram.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/A2A-Diagram.8666928d648acc26.webp" -o "$OUTPUT_DIR/lesson-11-a2a-diagram.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/nlweb-diagram.c1e2390b310e5fe4.webp" -o "$OUTPUT_DIR/lesson-11-nlweb-diagram.webp"

# Lesson 12 - Contexto
echo "📁 Lesson 12 - Contexto"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-12-thumbnail.ed19c94463e774d4.webp" -o "$OUTPUT_DIR/lesson-12-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/context-types.fc10b8927ee43f06.webp" -o "$OUTPUT_DIR/lesson-12-context-types.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/best-practices.f4170873dc554f58.webp" -o "$OUTPUT_DIR/lesson-12-best-practices.webp"

# Lesson 13 - Memoria
echo "📁 Lesson 13 - Memoria"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-13-thumbnail.959e3bc52d210c64.webp" -o "$OUTPUT_DIR/lesson-13-thumbnail.webp"

# Lesson 14 - MAF
echo "📁 Lesson 14 - MAF"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/lesson-14-thumbnail.90df0065b9d234ee.webp" -o "$OUTPUT_DIR/lesson-14-thumbnail.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/framework-intro.077af16617cf130c.webp" -o "$OUTPUT_DIR/lesson-14-framework-intro.webp"
curl -sL "https://raw.githubusercontent.com/microsoft/ai-agents-for-beginners/main/translated_images/es/agent-components.410a06daf87b4fef.webp" -o "$OUTPUT_DIR/lesson-14-agent-components.webp"

echo ""
echo "✅ Download complete!"
echo "📊 $(ls -1 $OUTPUT_DIR | wc -l) images downloaded to $OUTPUT_DIR"
echo ""
echo "📤 Next step: Upload to R2"
echo "   wrangler r2 object put yodev-assets/courses/ai-agents-es/images/ --file ./images/* --content-type image/webp"
