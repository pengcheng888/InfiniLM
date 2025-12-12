# C/C++ 代码跳转配置指南

由于 Cursor 中无法直接安装 Microsoft 的 C/C++ 扩展，我们使用 **clangd** 作为替代方案。

## 方案一：使用 clangd（推荐）

### 1. 安装 clangd

```bash
sudo apt install clangd
```

### 2. 在 Cursor 中安装 clangd 扩展

1. 按 `Ctrl+Shift+X` 打开扩展面板
2. 搜索 "clangd" 
3. 安装 **"clangd"** 扩展（作者：llvm-vs-code-extensions）

### 3. 禁用 Microsoft C/C++ 扩展（如果已安装）

如果之前安装了 Microsoft C/C++ 扩展，需要禁用它以避免冲突：
- 在扩展面板中找到 "C/C++" (Microsoft)
- 点击禁用或卸载

### 4. 重新加载窗口

- 按 `Ctrl+Shift+P`，输入 "Reload Window"
- 或重启 Cursor

### 5. 验证配置

- 打开任意 `.cpp` 或 `.hpp` 文件
- 按住 `Ctrl` 并点击函数名，应该能跳转到定义
- 或者右键点击函数名，选择 "Go to Definition"

## 方案二：手动安装 Microsoft C/C++ 扩展

如果必须使用 Microsoft 的扩展：

### 1. 下载扩展文件

访问：https://github.com/microsoft/vscode-cpptools/releases
下载最新版本的 `.vsix` 文件

### 2. 在 Cursor 中安装

1. 按 `Ctrl+Shift+X` 打开扩展面板
2. 点击右上角的 `...` 菜单
3. 选择 "Install from VSIX..."
4. 选择下载的 `.vsix` 文件

### 3. 更新配置

如果使用 Microsoft 扩展，需要将 `.vscode/c_cpp_properties.json` 中的：
```json
"configurationProvider": "llvm-vs-code-extensions.vscode-clangd"
```
改为：
```json
"configurationProvider": "ms-vscode.cpptools"
```

## 注意事项

- `compile_commands.json` 已经生成，clangd 会自动使用它
- 如果修改了项目结构，需要重新生成：`xmake project -k compile_commands`
- clangd 首次索引可能需要几分钟，请耐心等待


