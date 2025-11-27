import sharp from "sharp";
import type { TextureOpt } from "../types";
import { type GfxCtx, Texture } from "./gfx";

/**
 * @group Rendering
 * @subgroup Canvas
 */
export class FrameBuffer {
    ctx: GfxCtx;
    tex: Texture;
    glFramebuffer: WebGLFramebuffer;
    glRenderbuffer: WebGLRenderbuffer;
    _attached: boolean;

    constructor(ctx: GfxCtx, w: number, h: number, opt: TextureOpt = {}) {
        this.ctx = ctx;
        const gl = ctx.gl;
        ctx.onDestroy(() => this.free());
        this.tex = new Texture(ctx, w, h, opt);

        const frameBuffer = gl.createFramebuffer();
        const renderBuffer = gl.createRenderbuffer();

        if (!frameBuffer || !renderBuffer) {
            throw new Error("Failed to create framebuffer");
        }

        this.glFramebuffer = frameBuffer;
        this.glRenderbuffer = renderBuffer;

        // If texture storage hasn't been allocated yet (texture defers
        // allocation until it has valid size or data), postpone attaching
        // and storage until bind-time. We keep track if attachments are set.
        this._attached = false;
        if (this.tex._allocated) {
            this.bind();
            gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_STENCIL, w, h);
            gl.framebufferTexture2D(
                gl.FRAMEBUFFER,
                gl.COLOR_ATTACHMENT0,
                gl.TEXTURE_2D,
                this.tex.glTex,
                0,
            );
            gl.framebufferRenderbuffer(
                gl.FRAMEBUFFER,
                gl.DEPTH_STENCIL_ATTACHMENT,
                gl.RENDERBUFFER,
                this.glRenderbuffer,
            );
            // Check framebuffer completeness during development to catch issues
            const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
            if (status !== gl.FRAMEBUFFER_COMPLETE) {
                // eslint-disable-next-line no-console
                console.error("Framebuffer incomplete, status=", status);
            }
            this.unbind();
            this._attached = true;
        }
    }

    get width() {
        return this.tex.width;
    }

    get height() {
        return this.tex.height;
    }

    toImageData() {
        const gl = this.ctx.gl;
        const data = new Uint8ClampedArray(this.width * this.height * 4);
        this.bind();
        gl.readPixels(
            0,
            0,
            this.width,
            this.height,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            data,
        );
        this.unbind();
        // flip vertically
        const bytesPerRow = this.width * 4;
        const temp = new Uint8Array(bytesPerRow);
        for (let y = 0; y < (this.height / 2 | 0); y++) {
            const topOffset = y * bytesPerRow;
            const bottomOffset = (this.height - y - 1) * bytesPerRow;
            temp.set(data.subarray(topOffset, topOffset + bytesPerRow));
            data.copyWithin(
                topOffset,
                bottomOffset,
                bottomOffset + bytesPerRow,
            );
            data.set(temp, bottomOffset);
        }
        return new ImageData(data, this.width, this.height);
    }

    async toDataURL(): Promise<string> {
        const pngBuffer = await sharp(Buffer.from(this.toImageData().data), {
            raw: {
                width: this.width,
                height: this.height,
                channels: 4, // RGBA
            }
        }).png().toBuffer();
        
        const base64 = pngBuffer.toString('base64');
        return `data:image/png;base64,${base64}`;
    }

    clear() {
        const gl = this.ctx.gl;
        gl.clear(gl.COLOR_BUFFER_BIT);
    }

    draw(action: () => void) {
        this.bind();
        action();
        this.unbind();
    }

    bind() {
        const gl = this.ctx.gl;
        // Attach renderbuffer/texture storage if we deferred it earlier
        if (!this._attached) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.glFramebuffer);
            gl.bindRenderbuffer(gl.RENDERBUFFER, this.glRenderbuffer);
            gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_STENCIL, this.width, this.height);
            gl.framebufferTexture2D(
                gl.FRAMEBUFFER,
                gl.COLOR_ATTACHMENT0,
                gl.TEXTURE_2D,
                this.tex.glTex,
                0,
            );
            gl.framebufferRenderbuffer(
                gl.FRAMEBUFFER,
                gl.DEPTH_STENCIL_ATTACHMENT,
                gl.RENDERBUFFER,
                this.glRenderbuffer,
            );
            const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
            if (status !== gl.FRAMEBUFFER_COMPLETE) {
                // eslint-disable-next-line no-console
                console.error("Framebuffer incomplete on bind, status=", status);
            }
            // restore binds handled by push/pop
            this._attached = true;
            gl.bindFramebuffer(gl.FRAMEBUFFER, null as any);
            gl.bindRenderbuffer(gl.RENDERBUFFER, null as any);
        }

        this.ctx.pushFramebuffer(this.glFramebuffer as any);
        this.ctx.pushRenderbuffer(this.glRenderbuffer as any);
        this.ctx.pushViewport({ x: 0, y: 0, w: this.width, h: this.height });
    }

    unbind() {
        this.ctx.popFramebuffer();
        this.ctx.popRenderbuffer();
        this.ctx.popViewport();
    }

    free() {
        const gl = this.ctx.gl;
        gl.deleteFramebuffer(this.glFramebuffer);
        gl.deleteRenderbuffer(this.glRenderbuffer);
        this.tex.free();
    }
}
