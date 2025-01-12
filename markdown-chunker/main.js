import fs from 'node:fs'
import path from 'path'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { toMarkdown } from 'mdast-util-to-markdown'
import { u } from 'unist-builder'

const docsDir = path.join(import.meta.dirname, '../docs')

// split the contents by heading (max depth 2) and create chunks
const createChunks = (filePath) => {
  const chunksDir = path.join(docsDir, `chunks/${path.parse(filePath).name}`)
  if (!fs.existsSync(chunksDir)) {
    fs.mkdirSync(chunksDir, { recursive: true })
  } else {
    console.log(`Skipping... Chunks already exist at ${chunksDir}`)
    return
  }

  const content = fs.readFileSync(filePath)
  const tree = fromMarkdown(content)

  // since we are only considering headings till depth 2
  // we call depth one heading as parent.
  // Therefore heading with depth 2 will have a parent
  // and heading with depth one won't
  let parent = undefined
  tree.children = tree.children.reduce((tree, node) => {
    if (node.type === 'heading' && node.depth === 1) {
      tree.push(u('root', [node]))
      parent = node
    } else if (node.type === 'heading' && node.depth === 2) {
      // avoid depth one heading with no content before depth 2 heading
      const prevNodeChildren = tree.at(-1)?.children
      if (
        prevNodeChildren?.length === 1 &&
        prevNodeChildren[0].type === 'heading' &&
        prevNodeChildren[0].depth === 1
      )
        tree.pop()
      tree.push(u('root', [parent, node]))
    } else {
      tree.at(-1)?.children.push(node)
    }
    return tree
  }, [])

  tree.children.forEach((node, i) => {
    fs.writeFileSync(
      path.join(chunksDir, i.toString().padStart(3, 0)),
      toMarkdown(node),
    )
  })
}

const files = fs.readdirSync(docsDir).filter((x) => /.*\.md$/.test(x))
files.forEach((x) => createChunks(path.join(docsDir, x)))
