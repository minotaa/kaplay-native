import type { WebGLRenderingContext, WebGLTexture } from "@kmamal/gl";
import type { Shader, Uniform } from "../assets/shader";
import { IDENTITY_MATRIX } from "../constants/math";
import { getCamTransform } from "../game/camera";
import {
    BlendMode,
    type ImageSource,
    type KAPLAYOpt,
    type TextureOpt,
} from "../types";
import { deepEq } from "../utils/deepEq";
import type { Picture } from "./draw/drawPicture";

export type GfxCtx = ReturnType<typeof initGfx>;

/**
 * @group Rendering
 * @subgroup Canvas
 */
export class Texture {
    ctx: GfxCtx;
    src: null | ImageSource = null;
    glTex: WebGLTexture;
    width: number;
    height: number;
    _allocated: boolean;

    constructor(ctx: GfxCtx, w: number, h: number, opt: TextureOpt = {}) {
        this.ctx = ctx;

        const gl = ctx.gl;
        const glText = ctx.gl.createTexture();

        if (!glText) {
            throw new Error("[rendering] Failed to create texture");
        }

        this.glTex = glText;
        ctx.onDestroy(() => this.free());

        this.width = w;
        this.height = h;

        const filter = {
            "linear": gl.LINEAR,
            "nearest": gl.NEAREST,
        }[opt.filter ?? ctx.opts.texFilter ?? "nearest"];

        const wrap = {
            "repeat": gl.REPEAT,
            "clampToEdge": gl.CLAMP_TO_EDGE,
        }[opt.wrap ?? "clampToEdge"];

        this.bind();

        // If width/height are zero, defer allocating storage until we have
        // valid dimensions (this can happen if the window/context isn't ready
        // when textures are constructed). We mark allocated when storage is
        // created.
        this._allocated = false;
        if (w && h) {
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                w,
                h,
                0,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                null as any,
            );
            this._allocated = true;
        }

        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, 1);
        this.unbind();
    }

    static fromImage(
        ctx: GfxCtx,
        img: ImageSource,
        opt: TextureOpt = {},
    ): Texture {
        const tex = new Texture(ctx, img.width, img.height, opt);
        tex.update(img);
        tex.src = img;
        return tex;
    }

    update(img: ImageSource, x = 0, y = 0) {
        const gl = this.ctx.gl;
        this.bind();
        // Ensure storage exists before uploading
        if (!this._allocated) {
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA,
                img.width,
                img.height,
                0,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                null as any,
            );
            this._allocated = true;
        }
        const data = typeof img.data == 'function' ? img.data() : img.data;
        gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            x,
            y,
            img.width,
            img.height,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            data,
        );
        this.unbind();
    }

    bind() {
        this.ctx.pushTexture2D(this.glTex);
    }

    unbind() {
        this.ctx.popTexture2D();
    }

    /** Frees up texture memory. Call this once the texture is no longer being used to avoid memory leaks. */
    free() {
        this.ctx.gl.deleteTexture(this.glTex);
    }
}

/**
 * @group Rendering
 * @subgroup Shaders
 */
export type VertexFormat = {
    name: string;
    size: number;
}[];

/**
 * @group Rendering
 * @subgroup Canvas
 */
export class BatchRenderer {
    ctx: GfxCtx;

    glVBuf: WebGLBuffer;
    glIBuf: WebGLBuffer;
    vqueue: number[] = [];
    iqueue: number[] = [];
    stride: number;
    maxVertices: number;
    maxIndices: number;

    vertexFormat: VertexFormat;
    numDraws: number = 0;

    curPrimitive: GLenum | null = null;
    curTex: Texture | null = null;
    curShader: Shader | null = null;
    curUniform: Uniform | null = null;
    curBlend: BlendMode = BlendMode.Normal;
    curFixed: boolean | undefined = undefined;

    picture: Picture | null = null;

    constructor(
        ctx: GfxCtx,
        format: VertexFormat,
        maxVertices: number,
        maxIndices: number,
    ) {
        const gl = ctx.gl;

        this.vertexFormat = format;
        this.ctx = ctx;
        this.stride = format.reduce((sum, f) => sum + f.size, 0);
        this.maxVertices = maxVertices;
        this.maxIndices = maxIndices;

        console.log('Creating BatchRenderer with:', {
            maxVertices,
            maxIndices,
            vertexBufferSize: maxVertices * 4,
            indexBufferSize: maxIndices * 4
        });

        const glVBuf = gl.createBuffer();
        console.log('Buffer created:', glVBuf);

        if (!glVBuf) {
            throw new Error("Failed to create vertex buffer");
        }

        this.glVBuf = glVBuf;

        // Allocate and initialize the vertex buffer
        ctx.pushArrayBuffer(this.glVBuf);
        console.log('Buffer bound, about to allocate');
        gl.bufferData(gl.ARRAY_BUFFER, new ArrayBuffer(maxVertices * 4), gl.DYNAMIC_DRAW);
        let err = gl.getError();
        console.log('bufferData error:', err);
        ctx.popArrayBuffer();

        // Create and allocate the index buffer
        const glIBuf = gl.createBuffer();
        if (!glIBuf) {
            throw new Error("Failed to create index buffer");
        }
        this.glIBuf = glIBuf;
        ctx.pushElementArrayBuffer(this.glIBuf);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new ArrayBuffer(maxIndices * 2), gl.DYNAMIC_DRAW);
        ctx.popElementArrayBuffer();
    }

    push(
        primitive: GLenum,
        vertices: number[],
        indices: number[],
        shader: Shader,
        tex: Texture | null = null,
        uniform: Uniform | null = null,
        blend: BlendMode,
        width: number,
        height: number,
        fixed: boolean,
    ) {
        // If we have a picture, redirect data to the picture instead
        if (this.picture) {
            const index = this.picture.indices.length;
            const count = indices.length;
            const indexOffset = this.picture.vertices.length / this.stride;
            let l = vertices.length;
            for (let i = 0; i < l; i++) {
                this.picture.vertices.push(vertices[i]);
            }
            l = indices.length;
            for (let i = 0; i < l; i++) {
                this.picture.indices.push(indices[i] + indexOffset);
            }
            const material = {
                tex: tex || undefined,
                shader,
                uniform: uniform || undefined,
                blend,
            };
            if (this.picture.commands.length) {
                const lastCommand =
                    this.picture.commands[this.picture.commands.length - 1];
                const lastMaterial = lastCommand.material;
                if (
                    lastMaterial.tex == material.tex
                    && lastMaterial.shader == material.shader
                    && lastMaterial.uniform == material.uniform
                    && lastMaterial.blend == material.blend
                ) {
                    lastCommand.count += count;
                    return;
                }
            }
            const command = {
                material,
                index,
                count,
            };
            this.picture.commands.push(command);
            return;
        }

        // If texture, shader, blend mode or uniforms (including fixed) have changed, flush first
        // If the buffers are full, flush first
        if (
            primitive !== this.curPrimitive
            || tex !== this.curTex
            || shader !== this.curShader
            || ((this.curUniform != uniform)
                && !deepEq(this.curUniform, uniform))
            || blend !== this.curBlend
            || fixed !== this.curFixed
            // vertices is an array of floats already, so compare directly to maxVertices
            || this.vqueue.length + vertices.length > this.maxVertices
            || this.iqueue.length + indices.length > this.maxIndices
        ) {
            this.flush(width, height);
            this.setBlend(blend);
        }
        const indexOffset = this.vqueue.length / this.stride;
        let l = vertices.length;
        for (let i = 0; i < l; i++) {
            this.vqueue.push(vertices[i]);
        }
        l = indices.length;
        for (let i = 0; i < l; i++) {
            this.iqueue.push(indices[i] + indexOffset);
        }
        this.curPrimitive = primitive;
        this.curShader = shader;
        this.curTex = tex;
        this.curUniform = uniform;
        this.curFixed = fixed;
    }

    flush(width: number, height: number) {
        if (
            !this.curPrimitive
            || !this.curShader
            || this.vqueue.length === 0
            || this.iqueue.length === 0
        ) {
            return;
        }

        const gl = this.ctx.gl;

        let err
        this.ctx.pushArrayBuffer(this.glVBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(this.vqueue));
        err = gl.getError(); if (err) console.error('After bufferSubData vertex:', err);

        this.ctx.pushElementArrayBuffer(this.glIBuf);
        gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, new Uint16Array(this.iqueue));
        err = gl.getError(); if (err) console.error('After bufferSubData index:', err);

        this.ctx.setVertexFormat(this.vertexFormat);
        err = gl.getError(); if (err) console.error('After setVertexFormat:', err);

        this.curShader.bind();
        err = gl.getError(); if (err) console.error('After shader.bind:', err);

        if (this.curUniform) {
            this.curShader.send(this.curUniform);
            err = gl.getError(); if (err) console.error('After send uniform:', err);
        }

        this.curShader.send({
            width, height,
            camera: this.curFixed ? IDENTITY_MATRIX : getCamTransform(),
            transform: IDENTITY_MATRIX,
        });
        err = gl.getError(); if (err) console.error('After send system uniforms:', err);

        this.curTex?.bind();
        err = gl.getError(); if (err) console.error('After texture bind:', err);

        // Draw vertex buffer using active indices
        // Debug: check buffer sizes in case of potential overruns
        const arrayBufSize = gl.getBufferParameter
            ? gl.getBufferParameter(gl.ARRAY_BUFFER, gl.BUFFER_SIZE)
            : null;
        const elemBufSize = gl.getBufferParameter
            ? gl.getBufferParameter(gl.ELEMENT_ARRAY_BUFFER, gl.BUFFER_SIZE)
            : null;

        // Quick safety checks: ensure the GPU buffers are large enough
        const requiredArrayBytes = this.vqueue.length * 4; // floats -> bytes
        const requiredElemBytes = this.iqueue.length * 2; // uint16 -> bytes
        if ((arrayBufSize !== null && arrayBufSize < requiredArrayBytes)
            || (elemBufSize !== null && elemBufSize < requiredElemBytes)) {
            // eslint-disable-next-line no-console
            console.error("Insufficient GL buffer size before drawElements", {
                arrayBufSize,
                elemBufSize,
                requiredArrayBytes,
                requiredElemBytes,
                vqueueLen: this.vqueue.length,
                iqueueLen: this.iqueue.length,
            });
            // Avoid issuing draw call that will generate INVALID_OPERATION
            this.ctx.popArrayBuffer();
            this.ctx.popElementArrayBuffer();
            return;
        }

        gl.drawElements(
            this.curPrimitive,
            this.iqueue.length,
            gl.UNSIGNED_SHORT,
            0,
        );

        // Debug: check for GL errors immediately after issuing draw call
        err = gl.getError && gl.getError();
        if (err && err !== 0) {
            // Collect more debug info to help root cause
            const numVerts = this.vqueue.length / this.stride;
            const numIndices = this.iqueue.length;
            let maxIndex = -1;
            for (let i = 0; i < this.iqueue.length; i++) {
                if (this.iqueue[i] > maxIndex) maxIndex = this.iqueue[i];
            }
            const arrayBufBinding = gl.getParameter
                ? gl.getParameter(gl.ARRAY_BUFFER_BINDING)
                : null;
            const elemBufBinding = gl.getParameter
                ? gl.getParameter(gl.ELEMENT_ARRAY_BUFFER_BINDING)
                : null;
            const attrStates: any[] = [];
            try {
                for (let ai = 0; ai < this.vertexFormat.length; ai++) {
                    attrStates.push({
                        index: ai,
                        enabled: gl.getVertexAttrib(ai, gl.VERTEX_ATTRIB_ARRAY_ENABLED),
                        size: gl.getVertexAttrib(ai, gl.VERTEX_ATTRIB_ARRAY_SIZE),
                        stride: gl.getVertexAttrib(ai, gl.VERTEX_ATTRIB_ARRAY_STRIDE),
                        bufferBinding: gl.getVertexAttrib(ai, gl.VERTEX_ATTRIB_ARRAY_BUFFER_BINDING),
                        pointer: gl.getVertexAttribOffset ? gl.getVertexAttribOffset(ai, gl.VERTEX_ATTRIB_ARRAY_POINTER) : null,
                    });
                }
            } catch (e) {
                // some implementations might not support all queries; ignore
            }
            // eslint-disable-next-line no-console
            console.error("WebGL error after drawElements:", err, {
                primitive: this.curPrimitive,
                shader: this.curShader?.glProgram ?? this.curShader,
                tex: this.curTex?.glTex ?? this.curTex,
                numVerts,
                numIndices,
                stride: this.stride,
                maxIndex,
                vqueueLen: this.vqueue.length,
                iqueueLen: this.iqueue.length,
                firstVertices: this.vqueue.slice(0, Math.min(32, this.vqueue.length)),
                firstIndices: this.iqueue.slice(0, Math.min(32, this.iqueue.length)),
                arrayBufSize,
                elemBufSize,
                arrayBufBinding,
                elemBufBinding,
                attrStates,
                activeProgram: gl.getParameter ? gl.getParameter(gl.CURRENT_PROGRAM) : null,
            });
        }

        // Unbind texture and shader
        this.curTex?.unbind();
        this.curShader.unbind();

        // Unbind buffers
        this.ctx.popArrayBuffer();
        this.ctx.popElementArrayBuffer();

        // Reset local buffers
        this.vqueue.length = 0;
        this.iqueue.length = 0;

        // Increase draw
        this.numDraws++;
    }

    free() {
        const gl = this.ctx.gl;
        gl.deleteBuffer(this.glVBuf);
        gl.deleteBuffer(this.glIBuf);
    }

    setBlend(blend: BlendMode) {
        if (blend !== this.curBlend) {
            const gl = this.ctx.gl;
            this.curBlend = blend;
            switch (this.curBlend) {
                case BlendMode.Normal:
                    gl.blendFuncSeparate(
                        gl.ONE,
                        gl.ONE_MINUS_SRC_ALPHA,
                        gl.ONE,
                        gl.ONE_MINUS_SRC_ALPHA,
                    );
                    break;
                case BlendMode.Add:
                    gl.blendFuncSeparate(
                        gl.ONE,
                        gl.ONE,
                        gl.ONE,
                        gl.ONE_MINUS_SRC_ALPHA,
                    );
                    break;
                case BlendMode.Multiply:
                    gl.blendFuncSeparate(
                        gl.DST_COLOR,
                        gl.ZERO,
                        gl.ONE,
                        gl.ONE_MINUS_SRC_ALPHA,
                    );
                    break;
                case BlendMode.Screen:
                    gl.blendFuncSeparate(
                        gl.ONE_MINUS_DST_COLOR,
                        gl.ONE,
                        gl.ONE,
                        gl.ONE_MINUS_SRC_ALPHA,
                    );
                    break;
                case BlendMode.Overlay:
                    gl.blendFuncSeparate(
                        gl.DST_COLOR,
                        gl.ONE_MINUS_SRC_ALPHA,
                        gl.ONE,
                        gl.ONE_MINUS_SRC_ALPHA,
                    );
            }
        }
    }
}

/**
 * @group Rendering
 * @subgroup Shaders
 */
export class Mesh {
    ctx: GfxCtx;
    glVBuf: WebGLBuffer;
    glIBuf: WebGLBuffer;
    vertexFormat: VertexFormat;
    count: number;

    constructor(
        ctx: GfxCtx,
        format: VertexFormat,
        vertices: number[],
        indices: number[],
    ) {
        const gl = ctx.gl;
        this.vertexFormat = format;
        this.ctx = ctx;
        const glVBuf = gl.createBuffer();

        if (!glVBuf) throw new Error("Failed to create vertex buffer");

        this.glVBuf = glVBuf;

        ctx.pushArrayBuffer(this.glVBuf);
        gl.bufferData(
            gl.ARRAY_BUFFER,
            new Float32Array(vertices),
            gl.STATIC_DRAW,
        );
        ctx.popArrayBuffer();

        this.glIBuf = gl.createBuffer()!;
        ctx.pushElementArrayBuffer(this.glIBuf);
        gl.bufferData(
            gl.ELEMENT_ARRAY_BUFFER,
            new Uint16Array(indices),
            gl.STATIC_DRAW,
        );
        ctx.popElementArrayBuffer();

        this.count = indices.length;
    }

    draw(primitive?: GLenum, index?: GLuint, count?: GLuint): void {
        const gl = this.ctx.gl;
        this.ctx.pushArrayBuffer(this.glVBuf);
        this.ctx.pushElementArrayBuffer(this.glIBuf);
        this.ctx.setVertexFormat(this.vertexFormat);
        gl.drawElements(
            primitive ?? gl.TRIANGLES,
            count ?? this.count,      // count of indices
            gl.UNSIGNED_SHORT,
            index ?? 0,               // byte offset
        );
        this.ctx.popArrayBuffer();
        this.ctx.popElementArrayBuffer();
    }

    free() {
        const gl = this.ctx.gl;
        gl.deleteBuffer(this.glVBuf);
        gl.deleteBuffer(this.glIBuf);
    }
}

function genStack<T>(setFunc: (item: T | null) => void) {
    const stack: T[] = [];
    // TODO: don't do anything if pushed item is the same as the one on top?
    const push = (item: T) => {
        stack.push(item);
        setFunc(item);
    };
    const pop = () => {
        stack.pop();
        setFunc(cur() ?? null);
    };
    const cur = () => stack[stack.length - 1];
    return [push, pop, cur] as const;
}

export function initGfx(gl: WebGLRenderingContext, opts: KAPLAYOpt = {}) {
    const gc: Array<() => void> = [];

    function onDestroy(action: () => unknown) {
        gc.push(action);
    }

    function destroy() {
        gc.forEach((action) => action());
        // const extension = gl.getExtension("WEBGL_lose_context");
        // if (extension) extension.loseContext();
    }

    let curVertexFormat: object | null = null;

    function setVertexFormat(fmt: VertexFormat) {
        if (deepEq(fmt, curVertexFormat)) return;
        curVertexFormat = fmt;
        const stride = fmt.reduce((sum, f) => sum + f.size, 0);
        
        // Disable all attributes first
        const maxAttribs = gl.getParameter(gl.MAX_VERTEX_ATTRIBS);
        for (let i = fmt.length; i < maxAttribs; i++) {
            gl.disableVertexAttribArray(i);
        }
        
        // Now set up the current format
        fmt.reduce((offset, f, i) => {
            gl.enableVertexAttribArray(i);
            gl.vertexAttribPointer(
                i,
                f.size,
                gl.FLOAT,
                false,
                stride * 4,
                offset,
            );
            return offset + f.size * 4;
        }, 0);
    }

    const [pushTexture2D, popTexture2D] = genStack<WebGLTexture | null>((t) => {
        gl.bindTexture(gl.TEXTURE_2D, t || gl.createTexture()!); // Bind dummy if null
    });

    const [pushArrayBuffer, popArrayBuffer] = genStack<WebGLBuffer | null>((b) => {
        // @kmamal/gl might not support null, so we need to track differently
        // For now, just always bind - never unbind
        if (b) gl.bindBuffer(gl.ARRAY_BUFFER, b);
    });

    const [pushElementArrayBuffer, popElementArrayBuffer] = genStack<WebGLBuffer | null>((b) => {
        if (b) gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b);
    });

    const [pushFramebuffer, popFramebuffer] = genStack<WebGLFramebuffer | null>((b) => {
        if (b) gl.bindFramebuffer(gl.FRAMEBUFFER, b);
    });

    const [pushRenderbuffer, popRenderbuffer] = genStack<WebGLRenderbuffer | null>((b) => {
        if (b) gl.bindRenderbuffer(gl.RENDERBUFFER, b);
    });

    const [pushViewport, popViewport] = genStack<{ x: number; y: number; w: number; h: number } | null>((stack) => {
        if (!stack) return;
        const { x, y, w, h } = stack;
        gl.viewport(x, y, w, h);
    });

    const [pushProgram, popProgram] = genStack<WebGLProgram | null>((p) => {
        if (p) gl.useProgram(p);
    });

    pushViewport({
        x: 0,
        y: 0,
        w: gl.drawingBufferWidth,
        h: gl.drawingBufferHeight,
    });

    return {
        gl,
        opts,
        onDestroy,
        destroy,
        pushTexture2D,
        popTexture2D,
        pushArrayBuffer,
        popArrayBuffer,
        pushElementArrayBuffer,
        popElementArrayBuffer,
        pushFramebuffer,
        popFramebuffer,
        pushRenderbuffer,
        popRenderbuffer,
        pushViewport,
        popViewport,
        pushProgram,
        popProgram,
        setVertexFormat,
    };
}

// Debug helper: draw a simple triangle using attribute 0 only. Runs when
// opts.debugDrawTest is true. Useful to check if basic GL draw calls work
// outside the batcher and to validate attribute 0 behavior.
export function debugDrawTest(ggl: ReturnType<typeof initGfx>) {
    const gl = ggl.gl;
    try {
        // Simple passthrough vertex shader (uses attribute 0 as position)
        const vsrc = `attribute vec2 a_pos;void main(){gl_Position=vec4(a_pos,0.0,1.0);}`;
        const fsrc = `precision mediump float;void main(){gl_FragColor=vec4(1.0,0.0,1.0,1.0);}`;
        const prog = gl.createProgram();
        const vs = gl.createShader(gl.VERTEX_SHADER)!;
        const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
        gl.shaderSource(vs, vsrc);
        gl.shaderSource(fs, fsrc);
        gl.compileShader(vs);
        gl.compileShader(fs);
        gl.attachShader(prog!, vs);
        gl.attachShader(prog!, fs);
        // bind attrib 0 to a_pos
        gl.bindAttribLocation(prog!, 0, "a_pos");
        gl.linkProgram(prog!);
        gl.validateProgram(prog!);

        let err = gl.getError();
        console.log("debugDrawTest: after linkProgram glError=", err);

        const vbuf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbuf);
        err = gl.getError();
        console.log("debugDrawTest: after bindBuffer glError=", err);

        const verts = new Float32Array([0,0.5, -0.5,-0.5, 0.5,-0.5]);
        gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
        err = gl.getError();
        console.log("debugDrawTest: after bufferData glError=", err);

        gl.useProgram(prog!);
        err = gl.getError();
        console.log("debugDrawTest: after useProgram glError=", err);

        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        err = gl.getError();
        console.log("debugDrawTest: after vertexAttribPointer glError=", err);

        gl.clearColor(1,0,1,1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        err = gl.getError();
        console.log("debugDrawTest: after clear glError=", err);

        gl.drawArrays(gl.TRIANGLES, 0, 3);
        err = gl.getError();
        console.log("debugDrawTest: after drawArrays glError=", err);

        // cleanup
        gl.disableVertexAttribArray(0);
        gl.bindBuffer(gl.ARRAY_BUFFER, null as any);
        gl.useProgram(null as any);
        gl.deleteProgram(prog!);
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        gl.deleteBuffer(vbuf!);
    } catch (e) {
        console.error("debugDrawTest: exception", e);
    }
}