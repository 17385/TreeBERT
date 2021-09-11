from tree_sitter import Language, Parser

Language.build_library(
  'build/my-languages.so',
  [
    '/root/test_tree-sitter/tree-sitter-c-sharp',
    '/root/test_tree-sitter/tree-sitter-java',
    '/root/test_tree-sitter/tree-sitter-python'
  ]
)